"""
Federated learning algorithms operating on cached frozen-encoder features.

Implemented
-----------
Local-only    no communication; one model per client (lower bound)
Centralized   pooled data, one model (privacy-violating upper bound)
FedAvg        McMahan et al., AISTATS'17
FedProx       Li et al., MLSys'20            (proximal term)
FedPer        Arivazhagan et al., 2019       (private classifier layer)
pFedMe        T. Dinh et al., NeurIPS'20     (Moreau-envelope personalisation)
FedGIM        PROPOSED - FedProx proximal + FedPer private head
                         + class-balanced effective-number loss

Every method trains the identical `AdapterHead` topology on the identical
cached features, so the comparison isolates the aggregation strategy.
Communication is accounted honestly: only parameters actually transmitted are
counted, which is why FedPer / FedGIM show a lower per-round payload.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CFG, N_CLASSES
from .models import AdapterHead, ClassBalancedLoss, clone_state, make_head

PERSONALIZED = {"Local-only", "FedPer", "pFedMe", "FedGIM"}
GLOBAL_METHODS = {"Centralized", "FedAvg", "FedProx"}

# How much of the model is client-specific decides how a checkpoint is loaded.
# FedPer / FedGIM keep only a private suffix and inherit the rest from the
# server.  Local-only and pFedMe are personalised in their entirety - pFedMe's
# theta IS the deployed model - so merging them against `private=()` would
# silently return the global weights and erase the personalisation.
FULL_PERSONAL = {"Local-only", "pFedMe"}


def personal_head(method: str, in_dim: int, global_state: dict,
                  client_state: dict) -> "AdapterHead":
    """Rebuild one client's deployed model, respecting its personalisation."""
    if method in FULL_PERSONAL:
        h = make_head(in_dim, N_CLASSES)
        h.load_state_dict(client_state)
        h.eval()
        return h
    private = (CFG.fed.fedper_private_suffix
               if method in ("FedPer", "FedGIM") else ())
    h = _load(in_dim, global_state, client_state, private)
    h.eval()
    return h


# ---------------------------------------------------------------------------
@dataclass
class ClientData:
    cid: int
    Z: torch.Tensor          # (n, V, D) cached multi-view features
    y: torch.Tensor          # (n,) multi-class label
    ybin: torch.Tensor       # (n,) binary screening label

    @property
    def n(self) -> int:
        return int(self.Z.shape[0])

    def class_counts(self, n_classes: int = N_CLASSES) -> np.ndarray:
        return np.bincount(self.y.numpy(), minlength=n_classes)

    def batches(self, batch_size: int, generator: torch.Generator):
        """One epoch of shuffled minibatches, each drawing a random view."""
        perm = torch.randperm(self.n, generator=generator)
        V = self.Z.shape[1]
        for i in range(0, self.n, batch_size):
            idx = perm[i:i + batch_size]
            if len(idx) < 2:          # BatchNorm needs >1 sample
                continue
            v = torch.randint(0, V, (1,), generator=generator).item()
            yield self.Z[idx, v, :], self.y[idx]


@dataclass
class FLResult:
    method: str
    seed: int
    alpha: float
    is_personalized: bool
    global_state: dict | None = None
    client_states: list[dict] = field(default_factory=list)
    history: pd.DataFrame | None = None
    in_dim: int = 0
    n_classes: int = N_CLASSES
    comm_mb_total: float = 0.0

    def build_head(self, state: dict) -> AdapterHead:
        h = make_head(self.in_dim, self.n_classes)
        h.load_state_dict(state)
        h.eval()
        return h

    def heads(self) -> list[AdapterHead]:
        if self.is_personalized:
            return [self.build_head(s) for s in self.client_states]
        return [self.build_head(self.global_state)]


# ---------------------------------------------------------------------------
# Local optimisation primitives
# ---------------------------------------------------------------------------
def _local_train(head: AdapterHead,
                 client: ClientData,
                 criterion: nn.Module,
                 epochs: int,
                 lr: float,
                 generator: torch.Generator,
                 global_state: dict | None = None,
                 mu: float = 0.0) -> float:
    """Standard local SGD. `mu > 0` adds the FedProx proximal penalty."""
    head.train()
    opt = torch.optim.AdamW(head.parameters(), lr=lr,
                            weight_decay=CFG.fed.weight_decay)
    anchor = None
    if mu > 0 and global_state is not None:
        anchor = {k: v.detach().clone() for k, v in global_state.items()
                  if torch.is_floating_point(v)}

    total, nb = 0.0, 0
    for _ in range(epochs):
        for zb, yb in client.batches(CFG.fed.batch_size, generator):
            opt.zero_grad(set_to_none=True)
            loss = criterion(head(zb), yb)
            if anchor is not None:
                prox = sum(((p - anchor[k]) ** 2).sum()
                           for k, p in head.named_parameters()
                           if k in anchor)
                loss = loss + 0.5 * mu * prox
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
            opt.step()
            total += float(loss.detach()); nb += 1
    return total / max(nb, 1)


def _pfedme_local(head: AdapterHead,
                  client: ClientData,
                  criterion: nn.Module,
                  epochs: int,
                  lr: float,
                  generator: torch.Generator) -> tuple[float, dict]:
    """pFedMe: K inner steps solve the Moreau envelope, then w moves toward it.

        theta_i ~ argmin_t  f_i(t) + (lambda/2)||t - w_i||^2
        w_i     <- w_i - eta * lambda * (w_i - theta_i)

    Returns the mean loss *and* theta - the personalised model.  theta is what
    pFedMe actually deploys; returning only w would evaluate the method on its
    local copy of the global model and understate it badly.

    The inner solve uses the same AdamW settings as every other method's local
    optimiser, so the benchmark compares aggregation strategies rather than
    optimisers.  A fresh optimiser per batch keeps the K-step solve a genuine
    restart from w, as the algorithm specifies.
    """
    lam, K = CFG.fed.pfedme_lambda, CFG.fed.pfedme_k_steps
    eta = CFG.fed.pfedme_outer_lr
    head.train()
    theta = AdapterHead(head.in_dim, head.n_classes)
    theta.train()
    total, nb = 0.0, 0
    for _ in range(epochs):
        for zb, yb in client.batches(CFG.fed.batch_size, generator):
            w = {k: v.detach().clone() for k, v in head.state_dict().items()}
            theta.load_state_dict(w)
            wp = {k: w[k] for k, _ in theta.named_parameters()}
            inner = torch.optim.AdamW(theta.parameters(), lr=lr,
                                      weight_decay=CFG.fed.weight_decay)
            for _ in range(K):
                inner.zero_grad(set_to_none=True)
                loss = criterion(theta(zb), yb)
                reg = sum(((p - wp[k]) ** 2).sum()
                          for k, p in theta.named_parameters())
                (loss + 0.5 * lam * reg).backward()
                torch.nn.utils.clip_grad_norm_(theta.parameters(), 5.0)
                inner.step()
            with torch.no_grad():
                th = theta.state_dict()
                for k, v in head.state_dict().items():
                    if torch.is_floating_point(v):
                        v.add_(-eta * lam * (v - th[k]))
            total += float(loss.detach()); nb += 1
    return total / max(nb, 1), clone_state(theta.state_dict())


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def _is_private(key: str, private_suffix) -> bool:
    return any(key.startswith(s + ".") or key == s for s in private_suffix)


def _fedavg_aggregate(states: list[dict], weights: np.ndarray,
                      private_suffix=()) -> dict:
    """Sample-count-weighted parameter average; private keys are skipped."""
    w = torch.as_tensor(weights, dtype=torch.float32)
    w = w / w.sum()
    out = {}
    for k, v in states[0].items():
        if _is_private(k, private_suffix):
            continue
        if not torch.is_floating_point(v):
            out[k] = v.clone()          # e.g. BN num_batches_tracked
            continue
        acc = torch.zeros_like(v, dtype=torch.float32)
        for wi, s in zip(w, states):
            acc += wi * s[k].to(torch.float32)
        out[k] = acc.to(v.dtype)
    return out


def _payload_bytes(state: dict, private_suffix=()) -> int:
    """Bytes one client uploads per round (float32 on the wire)."""
    return int(sum(v.numel() * 4 for k, v in state.items()
                   if torch.is_floating_point(v)
                   and not _is_private(k, private_suffix)))


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------
def train_federated(method: str,
                    clients: list[ClientData],
                    in_dim: int,
                    seed: int,
                    alpha: float,
                    val_Z: torch.Tensor | None = None,
                    val_ybin: torch.Tensor | None = None,
                    rounds: int | None = None,
                    verbose_every: int = 5,
                    logger=None) -> FLResult:
    from sklearn.metrics import roc_auc_score
    from .config import ABNORMAL_CLASSES, CLASS_TO_IDX

    rounds = rounds or CFG.fed.rounds
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)

    abn_idx = [CLASS_TO_IDX[c] for c in ABNORMAL_CLASSES]
    n_i = np.array([c.n for c in clients], dtype=np.float64)

    # --- loss ------------------------------------------------------------
    def make_criterion(counts=None) -> nn.Module:
        if method == "FedGIM":
            return ClassBalancedLoss(counts, CFG.fed.cb_beta, N_CLASSES)
        return nn.CrossEntropyLoss()

    private = CFG.fed.fedper_private_suffix if method in ("FedPer", "FedGIM") else ()
    mu = CFG.fed.mu_prox if method in ("FedProx", "FedGIM") else 0.0
    personalized = method in PERSONALIZED

    # --- Centralized: pool every client's data ---------------------------
    if method == "Centralized":
        pooled = ClientData(
            cid=-1,
            Z=torch.cat([c.Z for c in clients]),
            y=torch.cat([c.y for c in clients]),
            ybin=torch.cat([c.ybin for c in clients]),
        )
        head = make_head(in_dim, N_CLASSES)
        crit = make_criterion(pooled.class_counts())
        rows = []
        for r in range(rounds):
            loss = _local_train(head, pooled, crit, CFG.fed.local_epochs,
                                CFG.fed.lr, gen)
            auc = _val_auc(head, val_Z, val_ybin, abn_idx, roc_auc_score)
            rows.append({"round": r + 1, "train_loss": loss, "val_auc": auc,
                         "comm_mb_round": 0.0, "comm_mb_cum": 0.0})
            if logger and (r + 1) % verbose_every == 0:
                logger.info("  [Centralized] r%02d loss=%.4f val_auc=%.4f",
                            r + 1, loss, auc)
        return FLResult(method, seed, alpha, False,
                        global_state=clone_state(head.state_dict()),
                        history=pd.DataFrame(rows), in_dim=in_dim,
                        comm_mb_total=0.0)

    # --- Local-only: no communication at all -----------------------------
    if method == "Local-only":
        heads = [make_head(in_dim, N_CLASSES) for _ in clients]
        crits = [make_criterion(c.class_counts()) for c in clients]
        rows = []
        for r in range(rounds):
            losses = [_local_train(h, c, cr, CFG.fed.local_epochs,
                                   CFG.fed.lr, gen)
                      for h, c, cr in zip(heads, clients, crits)]
            auc = float(np.mean([
                _val_auc(h, val_Z, val_ybin, abn_idx, roc_auc_score)
                for h in heads]))
            rows.append({"round": r + 1,
                         "train_loss": float(np.average(losses, weights=n_i)),
                         "val_auc": auc, "comm_mb_round": 0.0,
                         "comm_mb_cum": 0.0})
            if logger and (r + 1) % verbose_every == 0:
                logger.info("  [Local-only] r%02d loss=%.4f val_auc=%.4f",
                            r + 1, rows[-1]["train_loss"], auc)
        return FLResult(method, seed, alpha, True,
                        client_states=[clone_state(h.state_dict()) for h in heads],
                        history=pd.DataFrame(rows), in_dim=in_dim,
                        comm_mb_total=0.0)

    # --- Genuinely federated methods -------------------------------------
    global_head = make_head(in_dim, N_CLASSES)
    global_state = clone_state(global_head.state_dict())
    # Personalised methods keep a persistent private copy per client.
    client_states = [clone_state(global_state) for _ in clients]
    crits = [make_criterion(c.class_counts()) for c in clients]

    payload = _payload_bytes(global_state, private)
    per_round_mb = payload * len(clients) * 2 / (1024 ** 2)   # up + down

    rows, comm_cum = [], 0.0
    for r in range(rounds):
        local_states, losses = [], []
        for i, c in enumerate(clients):
            head = make_head(in_dim, N_CLASSES)
            if method in ("FedPer", "FedGIM"):
                # shared body from the server, private head kept locally
                merged = clone_state(global_state)
                for k, v in client_states[i].items():
                    if _is_private(k, private):
                        merged[k] = v.clone()
                head.load_state_dict(merged)
            elif method == "pFedMe":
                head.load_state_dict(clone_state(global_state))
            else:
                head.load_state_dict(clone_state(global_state))

            theta_sd = None
            if method == "pFedMe":
                loss, theta_sd = _pfedme_local(
                    head, c, crits[i], CFG.fed.local_epochs, CFG.fed.lr, gen)
            else:
                loss = _local_train(head, c, crits[i], CFG.fed.local_epochs,
                                    CFG.fed.lr, gen, global_state, mu)
            sd = clone_state(head.state_dict())
            local_states.append(sd)          # w_i is what the server averages
            # ...but theta_i is what pFedMe deploys, so that is the client model.
            client_states[i] = theta_sd if theta_sd is not None else sd
            losses.append(loss)

        agg = _fedavg_aggregate(local_states, n_i, private)
        if method == "pFedMe":
            beta = CFG.fed.pfedme_beta
            for k in agg:
                if torch.is_floating_point(agg[k]):
                    agg[k] = ((1 - beta) * global_state[k].to(torch.float32)
                              + beta * agg[k].to(torch.float32)).to(agg[k].dtype)
        for k, v in agg.items():
            global_state[k] = v

        comm_cum += per_round_mb
        eval_head = make_head(in_dim, N_CLASSES)
        eval_head.load_state_dict(
            {**global_state,
             **{k: v for k, v in client_states[0].items()
                if _is_private(k, private)}})
        if personalized:
            auc = float(np.mean([
                _val_auc(_load(in_dim, global_state, client_states[i], private),
                         val_Z, val_ybin, abn_idx, roc_auc_score)
                for i in range(len(clients))]))
        else:
            auc = _val_auc(eval_head, val_Z, val_ybin, abn_idx, roc_auc_score)

        rows.append({"round": r + 1,
                     "train_loss": float(np.average(losses, weights=n_i)),
                     "val_auc": auc,
                     "comm_mb_round": per_round_mb,
                     "comm_mb_cum": comm_cum})
        if logger and (r + 1) % verbose_every == 0:
            logger.info("  [%s] r%02d loss=%.4f val_auc=%.4f comm=%.2f MB",
                        method, r + 1, rows[-1]["train_loss"], auc, comm_cum)

    return FLResult(method, seed, alpha, personalized,
                    global_state=global_state,
                    client_states=client_states if personalized else [],
                    history=pd.DataFrame(rows), in_dim=in_dim,
                    comm_mb_total=comm_cum)


def _load(in_dim: int, global_state: dict, client_state: dict,
          private) -> AdapterHead:
    h = make_head(in_dim, N_CLASSES)
    merged = clone_state(global_state)
    for k, v in client_state.items():
        if _is_private(k, private):
            merged[k] = v.clone()
    h.load_state_dict(merged)
    return h


@torch.no_grad()
def _val_auc(head: AdapterHead, Z, ybin, abn_idx, roc_auc_score) -> float:
    """Binary screening AUC on the validation set (view 0, no TTA)."""
    if Z is None:
        return float("nan")
    head.eval()
    p = F.softmax(head(Z[:, 0, :]), dim=1).numpy()
    score = p[:, abn_idx].sum(axis=1)
    y = ybin.numpy()
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


# ---------------------------------------------------------------------------
class ClientRouter:
    """Assigns an unseen image to a client model using image features only.

    Each centroid is the L2-normalised mean training embedding of one client,
    so routing never touches a label, a ground-truth client id, or anything
    derived from the test split.  This is what makes the pooled-test protocol
    honest: at deployment an incoming frame is scored by whichever site model
    best matches its appearance, exactly as it would be in the field.
    """

    def __init__(self, clients: list["ClientData"] | None = None,
                 view: int = 0, centroids: torch.Tensor | None = None):
        if centroids is not None:
            self.centroids = centroids
            return
        cs = []
        for c in clients:
            z = c.Z[:, view, :] if c.Z.dim() == 3 else c.Z
            m = z.mean(0)
            cs.append(m / (m.norm() + 1e-12))
        self.centroids = torch.stack(cs)                       # (K, D)

    @classmethod
    def from_centroids(cls, centroids: torch.Tensor) -> "ClientRouter":
        """Rebuild a router from a checkpoint without re-reading training data."""
        return cls(centroids=centroids)

    @torch.no_grad()
    def similarity(self, Z: torch.Tensor) -> torch.Tensor:
        Zn = Z / (Z.norm(dim=1, keepdim=True) + 1e-12)
        return Zn @ self.centroids.T                           # (N, K)

    @torch.no_grad()
    def assign(self, Z: torch.Tensor) -> np.ndarray:
        return self.similarity(Z).argmax(1).numpy()


@torch.no_grad()
def client_heads(result: FLResult) -> list:
    """Materialise one evaluation head per client (or the single global head)."""
    if not result.is_personalized:
        h = result.build_head(result.global_state)
        h.eval()
        return [h]
    return [personal_head(result.method, result.in_dim, result.global_state, st)
            for st in result.client_states]


@torch.no_grad()
def predict_test(result: FLResult,
                 test_Z: torch.Tensor,
                 router: "ClientRouter | None" = None,
                 view: int = 0,
                 mode: str = "route") -> np.ndarray:
    """Return (N, 6) logits for the whole pooled test set.

    Every method - global or personalised - predicts every held-out sample, so
    the comparison is apples-to-apples and the paired DeLong / Wilcoxon tests
    are valid.  Personalised methods need a rule for picking a client model on
    an unseen frame:

    ``route``     feature-space nearest-centroid routing (default, deployment
                  realistic; requires `router`)
    ``ensemble``  mean posterior over all client models (no routing decision)

    The test set is *not* partitioned by client - doing so would let a
    personalised model exploit its own client's label prior, which inflates
    AUC by up to 0.31 at alpha=0.5 with no image information at all.
    """
    Z = test_Z[:, view, :]
    heads = client_heads(result)
    if not result.is_personalized or len(heads) == 1:
        return heads[0](Z).numpy()

    if mode == "ensemble":
        p = np.mean([torch.softmax(h(Z), dim=1).numpy() for h in heads], axis=0)
        return np.log(np.clip(p, 1e-12, None))

    if router is None:
        raise ValueError("mode='route' needs a ClientRouter built from the "
                         "clients' training features")
    out = np.zeros((Z.shape[0], result.n_classes), dtype=np.float64)
    assign = router.assign(Z)
    for cid, head in enumerate(heads):
        m = assign == cid
        if m.any():
            out[m] = head(Z[torch.from_numpy(np.flatnonzero(m))]).numpy()
    return out
