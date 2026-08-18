"""
07 - Contribution C1: label-prior leakage in personalised-FL evaluation.

Evaluates the SAME trained models under two evaluation protocols:

  paired  train and test partitioned with one proportion matrix; each
          personalised model is scored on its own client's test shard
          (the prevailing practice this paper argues against)
  pooled  training partitioned, test held out whole; personalised models
          routed by image features alone (this paper's protocol)

Nothing is retrained - the protocols differ only in how the held-out set is
scored, which is precisely the point: an evaluation choice, not a modelling
choice, moves the reported numbers and the ranking.

Outputs
-------
    outputs/csv/protocol_comparison.csv      per method x seed, both protocols
    outputs/csv/prior_leak_alpha_sweep.csv   prior-only AUC over a fine alpha grid
    outputs/figures/fig15_protocol_inflation.{png,pdf}
    outputs/figures/fig16_rank_inversion.{png,pdf}
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

from src.config import (ABNORMAL_CLASSES, CACHE_DIR, CFG, CKPT_DIR,
                        CLASS_TO_IDX, CSV_DIR, FIG_DIR, METHODS, N_CLASSES)
from src.data_utils import dirichlet_partition_paired, dirichlet_proportions
from src.fl_algorithms import ClientRouter, client_heads
from src.models import make_head
from src.stats_utils import binary_screening_metrics
from src.utils import configure_torch, get_logger, set_seed

LOG = get_logger("07_leak")
plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update({"savefig.dpi": 300, "font.size": 9})
ABN = [CLASS_TO_IDX[c] for c in ABNORMAL_CLASSES]
MAIN_A = "0.5"
PALETTE = {"Local-only": "#9E9E9E", "Centralized": "#3C3C3C",
           "FedAvg": "#4C72B0", "FedProx": "#55A868", "FedPer": "#C44E52",
           "pFedMe": "#8172B2", "FedGIM": "#DD8452"}
PERSONALISED = {"Local-only", "FedPer", "pFedMe", "FedGIM"}


def rebuild(ck):
    """Heads + router + global head from a stage-03 checkpoint."""
    from src.fl_algorithms import personal_head
    in_dim = ck["in_dim"]
    if not ck["is_personalized"]:
        h = make_head(in_dim, N_CLASSES); h.load_state_dict(ck["global_state"])
        h.eval()
        return [h], None
    heads = [personal_head(ck["method"], in_dim, ck["global_state"], cs)
             for cs in ck["client_states"]]
    return heads, ClientRouter.from_centroids(ck["router_centroids"])


@torch.no_grad()
def score(heads, assign, Z):
    """Per-sample P(abnormal) given a head assignment."""
    out = np.zeros((Z.shape[0], N_CLASSES))
    if len(heads) == 1:
        out = torch.softmax(heads[0](Z), dim=1).numpy()
    else:
        for c, h in enumerate(heads):
            m = assign == c
            if m.any():
                idx = torch.from_numpy(np.flatnonzero(m))
                out[m] = torch.softmax(h(Z[idx]), dim=1).numpy()
    return out[:, ABN].sum(1)


def prior_only_auc(train_y, test_y, alpha, seed, paired: bool) -> float:
    """AUC of a predictor that sees no image - only the client base rate."""
    if paired:
        tr, te, _ = dirichlet_partition_paired(train_y, test_y,
                                               CFG.fed.n_clients, alpha, seed)
        s = np.zeros(len(test_y))
        for tri, tei in zip(tr, te):
            cnt = np.bincount(train_y[tri], minlength=N_CLASSES).astype(float)
            s[tei] = cnt[ABN].sum() / max(cnt.sum(), 1)
    else:
        cnt = np.bincount(train_y, minlength=N_CLASSES).astype(float)
        s = np.full(len(test_y), cnt[ABN].sum() / cnt.sum())
    s = s + np.random.RandomState(seed).normal(0, 1e-9, len(s))
    yb = (np.isin(test_y, ABN)).astype(int)
    return binary_screening_metrics(yb, s)["auc_roc"]


# ---------------------------------------------------------------------------
def main() -> None:
    configure_torch()
    feats = {s: torch.from_numpy(np.load(CACHE_DIR / f"feat_{s}.npy"))
             for s in ("train", "test")}
    mtr = pd.read_csv(CACHE_DIR / "meta_train.csv")
    mte = pd.read_csv(CACHE_DIR / "meta_test.csv")
    train_y, test_y = mtr.class_idx.to_numpy(), mte.class_idx.to_numpy()
    test_yb = mte.binary_label.to_numpy()
    Zt = feats["test"][:, 0, :]
    alpha = CFG.fed.main_alpha

    # ---- 1. same models, two protocols ------------------------------------
    rows, shard_match = [], []
    for seed in CFG.seeds:
        set_seed(seed)
        tr_p, te_p, _ = dirichlet_partition_paired(train_y, test_y,
                                                   CFG.fed.n_clients, alpha,
                                                   seed)
        # Ground-truth client id per test row - the paired protocol's assignment
        paired_assign = np.zeros(len(test_y), dtype=int)
        for c, idx in enumerate(te_p):
            paired_assign[idx] = c

        for meth in METHODS:
            p = CKPT_DIR / f"{meth}_s{seed}_a{MAIN_A}.pt"
            if not p.exists():
                LOG.warning("missing %s", p.name); continue
            ck = torch.load(p, map_location="cpu", weights_only=False)
            heads, router = rebuild(ck)

            # confirm the paired partitioner reproduces the training shards the
            # model was actually fitted on, so the two protocols are comparable
            saved = [np.asarray(s) for s in ck["train_shards"]]
            shard_match.append(all(np.array_equal(a, b)
                                   for a, b in zip(saved, tr_p)))

            pooled_assign = (router.assign(Zt) if router is not None
                             else np.zeros(len(test_y), dtype=int))
            s_pool = score(heads, pooled_assign, Zt)
            s_pair = score(heads, paired_assign, Zt)
            m_pool = binary_screening_metrics(test_yb, s_pool)
            m_pair = binary_screening_metrics(test_yb, s_pair)
            rows.append({
                "seed": seed, "method": meth,
                "personalised": meth in PERSONALISED,
                "auc_pooled": m_pool["auc_roc"], "auc_paired": m_pair["auc_roc"],
                "inflation": m_pair["auc_roc"] - m_pool["auc_roc"],
                "sens_pooled": m_pool["sensitivity_at_90spec"],
                "sens_paired": m_pair["sensitivity_at_90spec"],
            })
        LOG.info("seed %d done", seed)

    df = pd.DataFrame(rows)
    df.to_csv(CSV_DIR / "protocol_comparison.csv", index=False)
    LOG.info("training shards reproduced exactly: %d/%d",
             sum(shard_match), len(shard_match))

    g = (df.groupby("method")
         .agg(pooled=("auc_pooled", "mean"), pooled_sd=("auc_pooled", "std"),
              paired=("auc_paired", "mean"), paired_sd=("auc_paired", "std"),
              infl=("inflation", "mean"), infl_sd=("inflation", "std"),
              pers=("personalised", "first"))
         .sort_values("infl", ascending=False))
    LOG.info("\n%s", g.round(4).to_string())

    # ---- 2. does the ranking invert? --------------------------------------
    rk = pd.DataFrame({
        "pooled": df.groupby("method").auc_pooled.mean().rank(ascending=False),
        "paired": df.groupby("method").auc_paired.mean().rank(ascending=False)})
    rk["shift"] = rk.pooled - rk.paired
    LOG.info("rank under each protocol (1 = best):\n%s", rk.to_string())
    sp = rk.pooled.corr(rk.paired, method="spearman")
    LOG.info("Spearman rank correlation between protocols: %.3f", sp)

    # ---- 3. is the leak structural? fine alpha grid, no training -----------
    sweep = []
    grid = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0,
            float("inf")]
    for a in grid:
        for sd in range(20):
            sweep.append({"alpha": a,
                          "alpha_num": 1e9 if np.isinf(a) else a, "seed": sd,
                          "auc_paired": prior_only_auc(train_y, test_y, a, sd,
                                                       True),
                          "auc_pooled": prior_only_auc(train_y, test_y, a, sd,
                                                       False)})
    sw = pd.DataFrame(sweep)
    sw.to_csv(CSV_DIR / "prior_leak_alpha_sweep.csv", index=False)
    LOG.info("prior-only AUC vs alpha (20 seeds each):\n%s",
             sw.groupby("alpha")[["auc_paired", "auc_pooled"]].mean()
             .round(4).to_string())

    figures(g, df, rk, sw)
    LOG.info("STAGE 07 COMPLETE")


def figures(g, df, rk, sw):
    # --- Fig 15: per-method inflation -------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.4))
    ax = axes[0]
    y = np.arange(len(g))
    ax.barh(y, g["infl"], xerr=g["infl_sd"].fillna(0), capsize=3,
            color=["#C44E52" if p else "#4C72B0" for p in g["pers"]],
            edgecolor="black")
    ax.set_yticks(y); ax.set_yticklabels(g.index)
    ax.axvline(0, c="k", lw=1)
    ax.set_xlabel("AUROC inflation  (paired protocol - pooled protocol)")
    ax.set_title("(a) Who benefits from the leak", fontweight="bold")
    ax.grid(alpha=0.3, axis="x")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(fc="#C44E52", ec="k", label="personalised"),
                       Patch(fc="#4C72B0", ec="k", label="global")],
              frameon=False, fontsize=8, loc="lower right")

    ax = axes[1]
    for _, r in g.iterrows():
        ax.plot([0, 1], [r["pooled"], r["paired"]], "-o", ms=5, lw=1.8,
                color=PALETTE.get(r.name, "#777"), label=r.name)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["pooled\n(leak-free)", "paired\n(leaky)"])
    ax.set_ylabel("AUROC")
    ax.set_title("(b) Same models, two protocols", fontweight="bold")
    ax.legend(frameon=False, fontsize=7, ncol=2); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"fig15_protocol_inflation.{e}",
                    bbox_inches="tight")
    plt.close(fig)

    # --- Fig 16: rank inversion + structural sweep -------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.4))
    ax = axes[0]
    for m, r in rk.iterrows():
        ax.plot([0, 1], [r.pooled, r.paired], "-o", ms=6, lw=2,
                color=PALETTE.get(m, "#777"))
        ax.annotate(m, (1.02, r.paired), fontsize=8, va="center")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["pooled", "paired"])
    ax.set_ylabel("Rank (1 = best)")
    ax.invert_yaxis(); ax.set_xlim(-0.15, 1.55)
    ax.set_title("(a) The ranking inverts", fontweight="bold")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    s = sw.groupby("alpha_num")[["auc_paired", "auc_pooled"]].agg(["mean", "std"])
    x = np.arange(len(s))
    ax.errorbar(x, s[("auc_paired", "mean")], yerr=s[("auc_paired", "std")],
                marker="o", lw=2, capsize=3, color="#C44E52",
                label="paired protocol")
    ax.errorbar(x, s[("auc_pooled", "mean")], yerr=s[("auc_pooled", "std")],
                marker="s", lw=2, capsize=3, color="#55A868",
                label="pooled protocol")
    ax.axhline(0.5, ls="--", c="k", lw=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(["IID" if v >= 1e8 else f"{v:g}" for v in s.index],
                       rotation=45, fontsize=7)
    ax.set_xlabel("Dirichlet concentration alpha")
    ax.set_ylabel("Prior-only AUROC (no image)")
    ax.set_title("(b) Leak magnitude is structural, not dataset-specific",
                 fontweight="bold")
    ax.legend(frameon=False, fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"fig16_rank_inversion.{e}", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
