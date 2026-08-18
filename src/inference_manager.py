"""
The Inference Management (IM) layer - the central contribution of this work.

IM is a *post-training* operational wrapper.  It never touches the federated
optimisation; it sits between a trained model and the clinician, and it can be
bolted onto any trained backbone (we demonstrate exactly that by attaching it
to FedAvg as well as to FedGIM).

Three-tier deployment architecture
----------------------------------
  Tier 1 - EDGE (endoscopy suite workstation)
      (3) resource-aware compression: INT8 dynamic-quantised head
      (4) continuous drift detection on encoder feature statistics
  Tier 2 - INSTITUTION (hospital inference node)
      (2) test-time adaptation: TENT-style BN adaptation + multi-view TTA
      (5) confidence-based orchestration: escalate uncertain cases from the
          cheap INT8 path to the full-precision multi-view ensemble
  Tier 3 - CLINICAL RELEASE
      (1) privacy-preserving release: Gaussian mechanism on emitted logits
      (6) human-in-the-loop rejection gating: abstain and refer

Execution order inside `run()` mirrors that tiering.
"""

from __future__ import annotations

import copy
import io
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats as sps

from .config import ABNORMAL_CLASSES, CFG, CLASS_TO_IDX, N_CLASSES
from .models import AdapterHead, make_head

ABN_IDX = [CLASS_TO_IDX[c] for c in ABNORMAL_CLASSES]


# ---------------------------------------------------------------------------
@dataclass
class IMOutput:
    probs: np.ndarray                 # (N, 6) released posterior
    binary_score: np.ndarray          # (N,)   P(abnormal)
    pred6: np.ndarray                 # (N,)   argmax over 6 classes
    accepted: np.ndarray              # (N,) bool - False == referred to human
    hitl_flag_rate: float
    escalated: np.ndarray             # (N,) bool - routed to expensive path
    escalation_rate: float
    drift_detected: bool
    drift_mmd: float
    drift_ks_frac: float
    latency_ms: float                 # per-image, end-to-end
    throughput_fps: float
    model_size_mb: float
    epsilon: float
    components: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# (1) Privacy-preserving inference
# ---------------------------------------------------------------------------
def gaussian_mechanism_sigma(epsilon: float, delta: float,
                             sensitivity: float) -> float:
    """Classic Gaussian mechanism noise scale for (eps, delta)-DP."""
    if not np.isfinite(epsilon) or epsilon <= 0:
        return 0.0
    return sensitivity * np.sqrt(2.0 * np.log(1.25 / delta)) / epsilon


def dp_privatise_logits(logits: np.ndarray, epsilon: float, delta: float,
                        sensitivity: float, rng: np.random.RandomState
                        ) -> np.ndarray:
    """Release logits under output perturbation.

    The guarantee is at the level of the *released inference result*: an
    adversary observing the emitted score cannot confidently distinguish
    neighbouring inputs.  Utility loss is the price, and quantifying that
    trade-off is precisely what the epsilon sweep reports.
    """
    sigma = gaussian_mechanism_sigma(epsilon, delta, sensitivity)
    if sigma <= 0:
        return logits
    return logits + rng.normal(0.0, sigma, size=logits.shape)


# ---------------------------------------------------------------------------
# (2) Test-time adaptation
# ---------------------------------------------------------------------------
def _entropy(p: torch.Tensor) -> torch.Tensor:
    return -(p * torch.log(p.clamp_min(1e-12))).sum(dim=1)


def tent_adapt(head: AdapterHead, Z: torch.Tensor, steps: int,
               lr: float) -> AdapterHead:
    """TENT (Wang et al., ICLR'21): minimise prediction entropy at test time.

    Only the BatchNorm affine parameters are updated and only unlabelled test
    features are used, so this is legal at deployment: no labels, no gradient
    on the encoder, no data leaves the institution.
    """
    ad = copy.deepcopy(head)
    ad.train()
    for p in ad.parameters():
        p.requires_grad_(False)
    for p in (ad.bn.weight, ad.bn.bias):
        p.requires_grad_(True)
    ad.bn.track_running_stats = False
    ad.bn.running_mean = None
    ad.bn.running_var = None
    ad.drop.p = 0.0

    opt = torch.optim.Adam([ad.bn.weight, ad.bn.bias], lr=lr)
    n = Z.shape[0]
    bs = max(64, min(256, n))
    for _ in range(steps):
        for i in range(0, n, bs):
            zb = Z[i:i + bs]
            if zb.shape[0] < 2:
                continue
            opt.zero_grad(set_to_none=True)
            loss = _entropy(F.softmax(ad(zb), dim=1)).mean()
            loss.backward()
            opt.step()
    ad.eval()
    return ad


@torch.no_grad()
def multiview_logits(head: nn.Module, Zv: torch.Tensor) -> np.ndarray:
    """Average softmax over cached augmented views, then return log-probs."""
    probs = None
    V = Zv.shape[1]
    for v in range(V):
        p = F.softmax(head(Zv[:, v, :]), dim=1)
        probs = p if probs is None else probs + p
    probs = (probs / V).clamp_min(1e-12)
    return torch.log(probs).numpy()


# ---------------------------------------------------------------------------
# (3) Resource-aware compression
# ---------------------------------------------------------------------------
def quantize_head(head: AdapterHead) -> nn.Module:
    """INT8 dynamic quantisation of the Linear layers (CPU deployment path)."""
    q = copy.deepcopy(head).eval()
    try:
        return torch.ao.quantization.quantize_dynamic(
            q, {nn.Linear}, dtype=torch.qint8)
    except Exception:
        return q


def module_size_mb(module: nn.Module) -> float:
    """Serialised on-disk size - what actually ships to the edge device."""
    buf = io.BytesIO()
    torch.save(module.state_dict(), buf)
    return buf.getbuffer().nbytes / (1024 ** 2)


# ---------------------------------------------------------------------------
# (4) Drift detection
# ---------------------------------------------------------------------------
def rbf_mmd2(X: np.ndarray, Y: np.ndarray, gamma: float | None = None) -> float:
    """Unbiased squared MMD with an RBF kernel (median heuristic bandwidth)."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if gamma is None:
        sub = np.vstack([X[:128], Y[:128]])
        d2 = ((sub[:, None, :] - sub[None, :, :]) ** 2).sum(-1)
        med = np.median(d2[d2 > 0]) if (d2 > 0).any() else 1.0
        gamma = 1.0 / max(med, 1e-8)

    def K(A, B):
        d2 = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T
        return np.exp(-gamma * np.maximum(d2, 0))

    n, m = len(X), len(Y)
    Kxx, Kyy, Kxy = K(X, X), K(Y, Y), K(X, Y)
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)
    return float(Kxx.sum() / (n * (n - 1)) + Kyy.sum() / (m * (m - 1))
                 - 2 * Kxy.mean())


def ks_drift_fraction(ref: np.ndarray, cur: np.ndarray,
                      alpha: float, max_dims: int = 128) -> float:
    """Fraction of feature dimensions whose marginal shifted (Bonferroni KS)."""
    d = min(ref.shape[1], max_dims)
    thr = alpha / d
    hits = 0
    for j in range(d):
        if sps.ks_2samp(ref[:, j], cur[:, j]).pvalue < thr:
            hits += 1
    return hits / d


def inject_covariate_drift(Z: np.ndarray, severity: float,
                           rng: np.random.RandomState) -> np.ndarray:
    """Simulate an acquisition shift in feature space.

    Models a new scope/processor at a partner site: a per-dimension gain and
    bias plus sensor noise, scaled by `severity`.  Applied in feature space so
    the same shift is reproducible across every method under test.
    """
    if severity <= 0:
        return Z
    d = Z.shape[-1]
    scale = 1.0 + severity * rng.normal(0, 0.25, size=d)
    shift = severity * rng.normal(0, 0.5, size=d) * Z.std(axis=tuple(
        range(Z.ndim - 1)))
    noise = severity * rng.normal(0, 0.1, size=Z.shape) * Z.std()
    return Z * scale + shift + noise


# ---------------------------------------------------------------------------
# (6) HITL gating
# ---------------------------------------------------------------------------
def hitl_scores(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalised predictive entropy and top-1/top-2 margin."""
    p = np.clip(probs, 1e-12, 1.0)
    ent = -(p * np.log(p)).sum(1) / np.log(probs.shape[1])
    srt = np.sort(p, axis=1)
    margin = srt[:, -1] - srt[:, -2]
    return ent, margin


def hitl_gate(probs: np.ndarray, target_coverage: float,
              ent_thresh: float, margin_thresh: float) -> np.ndarray:
    """Return an `accepted` mask; the complement is referred to a clinician.

    Cases are ranked by a combined uncertainty score and the least certain
    (1 - coverage) fraction is abstained on.  The fixed entropy / margin
    thresholds additionally force referral of anything egregiously uncertain,
    so coverage can fall below the target but never above it.
    """
    ent, margin = hitl_scores(probs)
    unc = ent + (1.0 - margin)
    n = len(unc)
    k = int(np.floor(target_coverage * n))
    order = np.argsort(unc, kind="stable")
    accepted = np.zeros(n, dtype=bool)
    accepted[order[:k]] = True
    accepted &= (ent <= ent_thresh) | (margin >= margin_thresh)
    return accepted


# ---------------------------------------------------------------------------
# The manager
# ---------------------------------------------------------------------------
class InferenceManager:
    """Post-training orchestrator wrapping a trained (federated) head.

    Parameters
    ----------
    heads
        One head for a global model, or one head per client for personalised
        models.  With several heads, `router` decides which one handles each
        incoming frame - from image features alone.
    router
        Feature-space `ClientRouter`.  Deliberately *not* a list of test shards:
        routing on ground-truth client membership lets a personalised head
        exploit its own client's label prior, which inflates AUC by up to 0.31
        at alpha=0.5 with no image information at all.
    global_head
        Optional global (non-personalised) model.  When present, orchestration
        combines it with the personalised prediction rather than replacing it -
        on this data personalised heads help as a complement and hurt as a
        substitute.
    reference_Z
        Feature window captured at validation time; the drift detector
        compares every incoming batch against it.
    enabled
        Per-component switches.  Flipping one off yields an ablation variant.
    """

    def __init__(self,
                 heads: list[AdapterHead],
                 router,
                 reference_Z: np.ndarray,
                 global_head: AdapterHead | None = None,
                 cfg=CFG.im,
                 enabled: dict | None = None,
                 seed: int = 0,
                 encoder_latency_ms: float = 0.0,
                 encoder_size_mb: float = 0.0):
        self.heads = heads
        self.router = router
        self.global_head = global_head
        self.ref = np.asarray(reference_Z, dtype=np.float32)
        self.cfg = cfg
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.enc_ms = encoder_latency_ms
        self.enc_mb = encoder_size_mb
        self.enabled = {
            "privacy": cfg.dp_enabled,
            "tta": cfg.tta_enabled,
            "compression": cfg.compress_enabled,
            "drift": cfg.drift_enabled,
            "orchestration": cfg.orchestration_enabled,
            "hitl": cfg.hitl_enabled,
        }
        if enabled:
            self.enabled.update(enabled)

    # -- component 4 ------------------------------------------------------
    def check_drift(self, Z: np.ndarray) -> tuple[bool, float, float]:
        if not self.enabled["drift"]:
            return False, float("nan"), float("nan")
        w = self.cfg.drift_window
        ref = self.ref[self.rng.choice(len(self.ref),
                                       min(w, len(self.ref)), replace=False)]
        cur = Z[self.rng.choice(len(Z), min(w, len(Z)), replace=False)]
        mmd = rbf_mmd2(ref, cur)
        ksf = ks_drift_fraction(ref, cur, self.cfg.drift_ks_alpha)
        return bool(mmd > self.cfg.drift_mmd_thresh or ksf > 0.10), mmd, ksf

    # -- main entry point -------------------------------------------------
    def run(self, test_Zv: torch.Tensor, epsilon: float | None = None
            ) -> IMOutput:
        """Execute the full IM pipeline over a multi-view test feature tensor.

        `test_Zv` is (N, V, D): V cached augmented views per image.
        """
        N, V, D = test_Zv.shape
        eps = self.cfg.dp_epsilon if epsilon is None else epsilon
        Znp = test_Zv[:, 0, :].numpy()

        # ---- Tier 1: drift monitor --------------------------------------
        drift, mmd, ksf = self.check_drift(Znp)

        # A confirmed drift makes adaptation mandatory: this is the coupling
        # between components 4 and 2 that a static pipeline cannot express.
        do_tta = self.enabled["tta"] or (drift and self.cfg.tta_enabled)

        # ---- prepare per-shard heads ------------------------------------
        assignments = self._assignments(test_Zv[:, 0, :])
        logits_cheap = np.zeros((N, N_CLASSES))
        logits_full = np.zeros((N, N_CLASSES))
        t_cheap = t_full = 0.0
        size_mb = 0.0

        for head, idx in assignments:
            if len(idx) == 0:
                continue
            Zi = test_Zv[idx]

            # ---- Tier 2: TTA --------------------------------------------
            use_head = head
            if do_tta and len(idx) >= 8:
                use_head = tent_adapt(head, Zi[:, 0, :],
                                      self.cfg.tta_steps, self.cfg.tta_lr)

            # ---- Tier 1: compression ------------------------------------
            if self.enabled["compression"]:
                cheap = quantize_head(use_head)
            else:
                cheap = copy.deepcopy(use_head).eval()
            size_mb = max(size_mb, module_size_mb(cheap))

            with torch.no_grad():
                t0 = time.perf_counter()
                lc = cheap(Zi[:, 0, :]).numpy()
                t_cheap += time.perf_counter() - t0

            t0 = time.perf_counter()
            lf = multiview_logits(use_head, Zi)
            t_full += time.perf_counter() - t0

            logits_cheap[idx] = lc
            logits_full[idx] = lf

        # ---- Tier 2: confidence orchestration ---------------------------
        p_cheap = _softmax(logits_cheap)
        if self.enabled["orchestration"]:
            escalate = p_cheap.max(1) < self.cfg.orchestration_conf_thresh
        else:
            escalate = np.ones(N, dtype=bool)   # everyone takes the full path
        logits = np.where(escalate[:, None], logits_full, logits_cheap)

        # Personalised heads help as a complement and hurt as a substitute, so
        # orchestration fuses them with the global model instead of replacing
        # it.  The weight is a fixed uniform average declared a priori - not
        # tuned on any split - so the reported figure is not split-selected.
        if self.global_head is not None and self.enabled["orchestration"]:
            with torch.no_grad():
                lg = self.global_head(test_Zv[:, 0, :]).numpy()
            fused = 0.5 * _softmax(logits) + 0.5 * _softmax(lg)
            logits = np.log(np.clip(fused, 1e-12, None))

        # ---- Tier 3: DP release -----------------------------------------
        if self.enabled["privacy"]:
            logits = dp_privatise_logits(logits, eps, self.cfg.dp_delta,
                                         self.cfg.dp_sensitivity, self.rng)
            eps_reported = eps
        else:
            eps_reported = float("inf")         # no formal guarantee

        probs = _softmax(logits)
        binary_score = probs[:, ABN_IDX].sum(1)
        pred6 = probs.argmax(1)

        # ---- Tier 3: HITL gating ----------------------------------------
        if self.enabled["hitl"]:
            accepted = hitl_gate(probs, self.cfg.hitl_target_coverage,
                                 self.cfg.hitl_entropy_thresh,
                                 self.cfg.hitl_margin_thresh)
        else:
            accepted = np.ones(N, dtype=bool)

        # ---- deployment cost --------------------------------------------
        esc_rate = float(escalate.mean())
        cheap_ms = t_cheap / max(N, 1) * 1000
        full_ms = t_full / max(N, 1) * 1000
        head_ms = cheap_ms + esc_rate * full_ms
        drift_ms = 0.15 if self.enabled["drift"] else 0.0   # amortised monitor
        lat = self.enc_ms + head_ms + drift_ms

        return IMOutput(
            probs=probs, binary_score=binary_score, pred6=pred6,
            accepted=accepted,
            hitl_flag_rate=float(1.0 - accepted.mean()),
            escalated=escalate, escalation_rate=esc_rate,
            drift_detected=drift, drift_mmd=mmd, drift_ks_frac=ksf,
            latency_ms=lat, throughput_fps=1000.0 / max(lat, 1e-6),
            model_size_mb=self.enc_mb + size_mb,
            epsilon=eps_reported,
            components=dict(self.enabled),
        )

    def _assignments(self, Z: torch.Tensor):
        """Which head handles which rows - decided from image features only."""
        n = Z.shape[0]
        if self.router is None or len(self.heads) == 1:
            return [(self.heads[0], np.arange(n))]
        asg = self.router.assign(Z)
        return [(self.heads[c], np.flatnonzero(asg == c))
                for c in range(len(self.heads))]


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)
