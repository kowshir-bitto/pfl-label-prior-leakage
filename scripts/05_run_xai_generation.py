"""
05 - STEP 2: Grad-CAM++ attribution and calibration analysis.

Grad-CAM++ needs spatial gradients, so this stage is the one place the cached
feature bank cannot be used: images are pushed through the real
encoder -> head graph.  Freezing the encoder stops the optimiser, not autograd,
so the attributions are exactly as valid as for an end-to-end fine-tuned net.

What is compared
----------------
    baseline   the strongest global baseline (FedAvg), vanilla inference
    proposed   FedGIM under the full Inference Management layer

Because the GIED release ships image-level labels and no lesion masks, no
"ground-truth boundary" is drawn anywhere.  Attribution quality is instead
reported through mask-free descriptors that a reviewer can verify:

    energy concentration  mass inside the hottest 10% of pixels   (higher better)
    CAM entropy           diffuseness of the attribution          (lower better)
    border mass           leakage into the vignetted frame edge   (lower better)
    agreement (IoU)       overlap between the two models' supports

Calibration is reported as Expected Calibration Error over 10 equal-width
confidence bins, plus the reliability curve behind Fig 12.

Outputs
-------
    outputs/csv/xai_cam_metrics.csv
    outputs/csv/calibration_bins.csv
    outputs/figures/fig14_xai_qualitative.{png,pdf}
    outputs/figures/fig12_reliability_ece.{png,pdf}
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from src.config import (CACHE_DIR, CFG, CKPT_DIR, CLASSES, CSV_DIR, FIG_DIR,
                        N_CLASSES)
from src.data_utils import imread_unicode
from src.models import MultiEncoder, make_head
from src.stats_utils import expected_calibration_error, reliability_bins
from src.utils import configure_torch, get_logger, set_seed
from src.xai import (GradCAMPlusPlus, cam_agreement, cam_border_mass,
                     cam_entropy, cam_energy_concentration, overlay_cam,
                     to_tensor)

LOG = get_logger("05_xai")
MAIN_A = "0.5"
BASELINE = "FedAvg"
PROPOSED = "FedGIM"
N_PER_CLASS = 4          # rows in the qualitative grid
CAM_SAMPLE = 240         # images per model for the quantitative descriptors


# ---------------------------------------------------------------------------
class EndToEnd(nn.Module):
    """encoder -> head, so Grad-CAM++ can reach a convolutional layer."""

    def __init__(self, encoder: MultiEncoder, head: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


def load_head(method: str, seed: int, in_dim: int):
    """Rebuild the deployed model for `method`.

    Personalised methods have no single model, so the CAM is taken from the
    posterior the IM layer actually releases: the uniform fusion of the global
    head with the personalised ensemble.  A `HeadEnsemble` reproduces that as
    one nn.Module so autograd sees the same graph the clinician's score came
    from.
    """
    p = CKPT_DIR / f"{method}_s{seed}_a{MAIN_A}.pt"
    if not p.exists():
        return None
    ck = torch.load(p, map_location="cpu", weights_only=False)
    from src.fl_algorithms import personal_head

    g = make_head(in_dim, N_CLASSES)
    g.load_state_dict(ck["global_state"])
    g.eval()
    if not ck["is_personalized"]:
        return g

    heads = [personal_head(ck["method"], in_dim, ck["global_state"], cs)
             for cs in ck["client_states"]]
    return HeadEnsemble(g, heads)


class HeadEnsemble(nn.Module):
    """Uniform fusion of the global head with the personalised ensemble.

    Mirrors the release rule in `InferenceManager.run`, so the explanation
    corresponds to the score the system actually emits.
    """

    def __init__(self, global_head: nn.Module, heads: list[nn.Module]):
        super().__init__()
        self.global_head = global_head
        self.heads = nn.ModuleList(heads)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        pg = torch.softmax(self.global_head(z), dim=1)
        pp = torch.stack([torch.softmax(h(z), dim=1) for h in self.heads]).mean(0)
        return torch.log((0.5 * pg + 0.5 * pp).clamp_min(1e-12))


# ---------------------------------------------------------------------------
def cam_descriptors(model: nn.Module, target_layer, paths, labels, tag):
    """Grad-CAM++ over a sample of test images; returns per-image descriptors."""
    rows, cams = [], {}
    with GradCAMPlusPlus(model, target_layer) as cam_fn:
        for pth, lab in tqdm(list(zip(paths, labels)), desc=f"cam/{tag}",
                             ncols=88):
            img = imread_unicode(Path(pth))
            if img is None:
                continue
            x = to_tensor(img).unsqueeze(0)
            cam, pred = cam_fn(x)
            c = cam[0]
            cams[pth] = c
            rows.append({
                "model": tag, "path": pth, "true_class": CLASSES[lab],
                "pred_class": CLASSES[int(pred[0])],
                "correct": bool(int(pred[0]) == lab),
                "energy_top10": cam_energy_concentration(c, 0.10),
                "cam_entropy": cam_entropy(c),
                "border_mass": cam_border_mass(c),
            })
    return pd.DataFrame(rows), cams


# ---------------------------------------------------------------------------
def figure_qualitative(sel, cams_b, cams_p, out_stem):
    """Input | baseline CAM | proposed CAM | difference, one row per class."""
    rows = len(sel)
    fig, axes = plt.subplots(rows, 4, figsize=(11.0, 2.75 * rows))
    axes = np.atleast_2d(axes)
    titles = ["Input (preprocessed)", f"{BASELINE} Grad-CAM++",
              f"{PROPOSED}+IM Grad-CAM++", "Difference (proposed - baseline)"]

    for r, (cls, pth) in enumerate(sel):
        img = imread_unicode(Path(pth))
        cb, cp = cams_b[pth], cams_p[pth]
        axes[r, 0].imshow(img[:, :, ::-1])
        axes[r, 1].imshow(overlay_cam(img, cb))
        axes[r, 2].imshow(overlay_cam(img, cp))
        d = axes[r, 3].imshow(cp - cb, cmap="coolwarm", vmin=-1, vmax=1)
        axes[r, 0].set_ylabel(cls, fontsize=11, fontweight="bold")
        for c in range(4):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(titles[c], fontsize=10, fontweight="bold")
    fig.colorbar(d, ax=axes[:, 3].tolist(), fraction=0.025, pad=0.02)
    fig.suptitle("Grad-CAM++ attribution: baseline vs Inference-Managed model\n"
                 "(GIED ships image-level labels only - no lesion masks, so no "
                 "ground-truth boundary is drawn)", fontsize=11,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure_reliability(bins_df, ece_tbl, out_stem):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="perfect calibration")
    for tag, g in bins_df.groupby("model"):
        # Weight each seed's bin by its count: empty bins carry NaN accuracy
        # and must not drag the curve toward an unpopulated region.
        g = g[g["count"] > 0]
        gg = (g.assign(w=g["accuracy"] * g["count"])
              .groupby("bin_mid", as_index=False)
              .agg(w=("w", "sum"), n=("count", "sum")))
        gg["acc"] = gg.w / gg.n
        ax.plot(gg.bin_mid, gg.acc, "o-", lw=2, ms=5, label=tag)
    ax.set_xlabel("Confidence"); ax.set_ylabel("Empirical accuracy")
    ax.set_title("(a) Reliability diagram (10 bins)", fontweight="bold")
    ax.legend(frameon=False, fontsize=9); ax.grid(alpha=0.3)

    ax = axes[1]
    t = ece_tbl.sort_values("ece_mean")
    ax.barh(t.model, t.ece_mean, xerr=t.ece_std, color=["#4C72B0", "#DD8452"],
            edgecolor="black", capsize=4)
    for i, (_, r) in enumerate(t.iterrows()):
        ax.text(r.ece_mean + (r.ece_std or 0) + 0.002, i,
                f"{r.ece_mean:.4f}", va="center", fontsize=9)
    ax.set_xlabel("Expected Calibration Error (lower is better)")
    ax.set_title("(b) ECE over 5 seeds", fontweight="bold")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
def main() -> None:
    configure_torch()
    meta = pd.read_csv(CACHE_DIR / "meta_test.csv")
    y6 = meta.class_idx.to_numpy()
    in_dim = int(np.load(CACHE_DIR / "feat_test.npy", mmap_mode="r").shape[-1])

    encoder = MultiEncoder(CFG.model.backbones, pretrained=True).eval()
    # ResNet-18 layer4: the deepest layer whose spatial grid is still 7x7, and
    # the conventional Grad-CAM target.  Gradients reach it through the concat.
    target_layer = encoder.encoders[0].target_layer

    # ---- calibration across all seeds, on cached features ------------------
    feat_test = torch.from_numpy(np.load(CACHE_DIR / "feat_test.npy"))
    bin_rows, ece_rows = [], []
    for seed in CFG.seeds:
        for tag, meth in ((BASELINE, BASELINE), (f"{PROPOSED}+IM", PROPOSED)):
            head = load_head(meth, seed, in_dim)
            if head is None:
                LOG.warning("missing checkpoint %s s%d", meth, seed)
                continue
            with torch.no_grad():
                probs = torch.softmax(head(feat_test[:, 0, :]), dim=1).numpy()
            ece = expected_calibration_error(probs, y6, n_bins=10)
            ece_rows.append({"model": tag, "seed": seed, "ece": ece})
            rb = reliability_bins(probs, y6, n_bins=10).assign(model=tag,
                                                              seed=seed)
            bin_rows.append(rb)
            LOG.info("seed %-5d %-12s ECE=%.4f", seed, tag, ece)

    bins_df = pd.concat(bin_rows, ignore_index=True)
    ece_df = pd.DataFrame(ece_rows)
    bins_df.to_csv(CSV_DIR / "calibration_bins.csv", index=False)
    ece_tbl = (ece_df.groupby("model", as_index=False)
               .agg(ece_mean=("ece", "mean"), ece_std=("ece", "std")))
    LOG.info("ECE summary:\n%s", ece_tbl.to_string(index=False))

    # ---- Grad-CAM++ on a stratified image sample ---------------------------
    set_seed(CFG.seeds[0])
    rng = np.random.RandomState(CFG.seeds[0])
    per = max(1, CAM_SAMPLE // N_CLASSES)
    idx = np.concatenate([
        rng.choice(np.flatnonzero(y6 == c), min(per, int((y6 == c).sum())),
                   replace=False) for c in range(N_CLASSES)])
    paths = meta.proc_path.to_numpy()[idx]
    labs = y6[idx]

    seed0 = CFG.seeds[0]
    mb = EndToEnd(encoder, load_head(BASELINE, seed0, in_dim)).eval()
    mp = EndToEnd(encoder, load_head(PROPOSED, seed0, in_dim)).eval()
    df_b, cams_b = cam_descriptors(mb, target_layer, paths, labs, BASELINE)
    df_p, cams_p = cam_descriptors(mp, target_layer, paths, labs,
                                   f"{PROPOSED}+IM")

    cam_df = pd.concat([df_b, df_p], ignore_index=True)
    agree = [cam_agreement(cams_b[p], cams_p[p]) for p in cams_b if p in cams_p]
    cam_df.to_csv(CSV_DIR / "xai_cam_metrics.csv", index=False)

    summ = (cam_df.groupby("model")
            .agg(energy_top10=("energy_top10", "mean"),
                 cam_entropy=("cam_entropy", "mean"),
                 border_mass=("border_mass", "mean"),
                 accuracy=("correct", "mean")))
    LOG.info("CAM descriptors:\n%s", summ.to_string())
    LOG.info("mean baseline/proposed CAM agreement (IoU@0.5) = %.4f",
             float(np.mean(agree)) if agree else float("nan"))

    # ---- figures -----------------------------------------------------------
    sel = []
    for c in range(N_CLASSES):
        cand = [p for p, l in zip(paths, labs) if l == c and p in cams_b]
        if cand:
            sel.append((CLASSES[c], cand[0]))
    figure_qualitative(sel, cams_b, cams_p, str(FIG_DIR / "fig14_xai_qualitative"))
    figure_reliability(bins_df, ece_tbl, str(FIG_DIR / "fig12_reliability_ece"))
    LOG.info("STAGE 05 COMPLETE")


if __name__ == "__main__":
    main()
