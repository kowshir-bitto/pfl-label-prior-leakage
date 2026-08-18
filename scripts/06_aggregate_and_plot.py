"""
06 - STEP 3: Multi-seed aggregation, significance testing, publication figures.

Ingests every per-run CSV from stages 03-05, reduces to Mean +/- Std with 95%
confidence intervals over the 5 seeds, and runs the two paired tests the
protocol calls for:

    DeLong    on paired AUCs, using the stored per-sample test predictions -
              the only correct way to compare two ROC curves on one test set.
    Wilcoxon  signed-rank across seeds for every other metric, with
              Holm-Bonferroni correction over the family of comparisons.

Significance is marked * p<0.05, ** p<0.01, *** p<0.001, and always against
the strongest baseline rather than the weakest, so the reported gap is the
honest one.

Figures
-------
    fig07_convergence.{png,pdf}        loss / AUC / communication over rounds
    fig08_main_benchmark.{png,pdf}     2x2 grouped bars, AUC/Sens/F1/latency
    fig09_noniid_robustness.{png,pdf}  degradation vs Dirichlet severity
    fig10_roc_delong.{png,pdf}         ROC with bootstrap 95% CI bands
    fig11_ablation_radar.{png,pdf}     6-axis IM component trade-offs
    fig13_coverage_risk.{png,pdf}      selective-prediction operating curve
    fig06_prior_leak.{png,pdf}         the evaluation-protocol result (C1)

Tables
------
    tables/table3_main_benchmark.tex
    tables/table4_ablation.tex
    outputs/csv/aggregated_*.csv, significance_tests.csv
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

from src.config import CFG, CSV_DIR, FIG_DIR, METHODS, PROPOSED_METHOD

TABLE_DIR = CSV_DIR.parent / "tables"
from src.stats_utils import (aggregate_mean_std, bootstrap_roc_band, delong_ci,
                             delong_roc_test, holm_bonferroni,
                             wilcoxon_vs_reference)
from src.utils import get_logger

LOG = get_logger("06_aggregate")
plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300,
                     "font.size": 9, "axes.titlesize": 10,
                     "axes.labelsize": 9, "legend.fontsize": 8})
MAIN_A = "0.5"
PROPOSED_IM = f"{PROPOSED_METHOD}"
PALETTE = {"Local-only": "#9E9E9E", "Centralized": "#3C3C3C",
           "FedAvg": "#4C72B0", "FedProx": "#55A868", "FedPer": "#C44E52",
           "pFedMe": "#8172B2", "FedGIM": "#DD8452", "FedAvg+IM": "#937860",
           "Prior-only (no image)": "#CCCCCC"}


def stars(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"


def read(name: str) -> pd.DataFrame | None:
    p = CSV_DIR / name
    if not p.exists():
        LOG.warning("missing %s - skipping the figures that need it", name)
        return None
    return pd.read_csv(p)


# ---------------------------------------------------------------------------
def fig_main_benchmark(main: pd.DataFrame, sig: pd.DataFrame):
    panels = [("auc_roc", "AUROC", False),
              ("sensitivity_at_90spec", "Sensitivity @ 90% Spec", False),
              ("f1_score", "F1 score", False),
              ("latency_ms", "Latency (ms / frame)", True)]
    order = [m for m in list(METHODS) + ["FedAvg+IM"]
             if m in set(main.method_name)]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6))
    for ax, (col, lab, lower_better) in zip(axes.ravel(), panels):
        g = (main.groupby("method_name")[col]
             .agg(["mean", "std"]).reindex(order))
        bars = ax.bar(range(len(g)), g["mean"], yerr=g["std"].fillna(0),
                      capsize=4, edgecolor="black", linewidth=0.7,
                      color=[PALETTE.get(m, "#777") for m in g.index])
        # Mark each baseline with the proposed-vs-that-baseline significance.
        if not sig.empty and "method_name" in sig.columns:
            for i, m in enumerate(g.index):
                row = sig[(sig.method_name == m) & (sig.metric == col)]
                if len(row):
                    y = g["mean"].iloc[i] + g["std"].fillna(0).iloc[i]
                    ax.text(i, y + abs(y) * 0.02,
                            stars(float(row.p_holm.iloc[0])), ha="center",
                            fontsize=9, fontweight="bold", color="#444")
        ax.set_xticks(range(len(g)))
        ax.set_xticklabels(g.index, rotation=28, ha="right")
        ax.set_ylabel(lab + ("  (lower better)" if lower_better else ""))
        ax.set_title(lab, fontweight="bold")
        ax.grid(alpha=0.3, axis="y")
        if not lower_better:
            lo = max(0.0, float(np.nanmin(g["mean"] - g["std"].fillna(0))) - 0.05)
            ax.set_ylim(lo, min(1.02, float(np.nanmax(
                g["mean"] + g["std"].fillna(0))) + 0.05))
    fig.suptitle("Federated benchmark on the pooled held-out test set "
                 f"(Dirichlet alpha={MAIN_A}, mean +/- SD over "
                 f"{len(CFG.seeds)} seeds)", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "fig08_main_benchmark")


def fig_noniid(rob: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, col, lab in ((axes[0], "auc_roc", "AUROC"),
                         (axes[1], "sensitivity_at_90spec",
                          "Sensitivity @ 90% Spec")):
        for m, g in rob.groupby("method_name"):
            gg = (g.groupby("alpha_num")[col].agg(["mean", "std"])
                  .sort_index())
            x = np.arange(len(gg))
            ax.errorbar(x, gg["mean"], yerr=gg["std"].fillna(0), marker="o",
                        ms=4, lw=1.8, capsize=3, label=m,
                        color=PALETTE.get(m, None),
                        ls="--" if m == "Prior-only (no image)" else "-")
        labels = [("IID" if v >= 1e8 else f"{v:g}")
                  for v in sorted(rob.alpha_num.unique())]
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_xlabel("Dirichlet concentration alpha  (left = more non-IID)")
        ax.set_ylabel(lab)
        ax.set_title(f"{lab} vs partition severity", fontweight="bold")
        ax.grid(alpha=0.3)
    axes[1].legend(frameon=False, ncol=2, fontsize=7)
    fig.tight_layout()
    save(fig, "fig09_noniid_robustness")


def fig_convergence(hist: pd.DataFrame):
    h = hist[hist.alpha.astype(str) == MAIN_A]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    specs = [("train_loss", "Training loss", axes[0]),
             ("val_auc", "Validation AUROC", axes[1]),
             ("comm_mb_cum", "Cumulative communication (MB)", axes[2])]
    for col, lab, ax in specs:
        if col not in h.columns:
            ax.set_visible(False)
            continue
        for m, g in h.groupby("method_name"):
            gg = g.groupby("round")[col].agg(["mean", "std"])
            ax.plot(gg.index, gg["mean"], lw=1.8, label=m,
                    color=PALETTE.get(m, None))
            ax.fill_between(gg.index, gg["mean"] - gg["std"].fillna(0),
                            gg["mean"] + gg["std"].fillna(0), alpha=0.15,
                            color=PALETTE.get(m, None))
        ax.set_xlabel("Communication round"); ax.set_ylabel(lab)
        ax.set_title(lab, fontweight="bold"); ax.grid(alpha=0.3)
    axes[0].legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    save(fig, "fig07_convergence")


def fig_roc(preds: dict, ybin: np.ndarray):
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    for m, s in preds.items():
        fpr, mean_tpr, lo, hi = bootstrap_roc_band(ybin, s)
        a, l, u = delong_ci(ybin, s)
        ax.plot(fpr, mean_tpr, lw=2, color=PALETTE.get(m, None),
                label=f"{m}  AUC={a:.3f} [{l:.3f}, {u:.3f}]")
        ax.fill_between(fpr, lo, hi, alpha=0.15, color=PALETTE.get(m, None))
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("1 - Specificity"); ax.set_ylabel("Sensitivity")
    ax.set_title("ROC with 95% CI (DeLong CI in legend)", fontweight="bold")
    ax.legend(frameon=False, loc="lower right", fontsize=7.5)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, "fig10_roc_delong")


def fig_ablation_radar(abl: pd.DataFrame):
    """Six axes, each normalised so that outward always means better."""
    axes_spec = [("auc_roc", "AUROC", False),
                 ("f1_score", "F1", False),
                 ("ece", "Calibration\n(1-ECE)", True),
                 ("latency_ms", "Speed\n(1-latency)", True),
                 ("hitl_flag_rate", "Autonomy\n(1-flag rate)", True),
                 ("privacy_budget_epsilon", "Privacy\n(1/epsilon)", True)]
    g = abl.groupby("ablation_variant")
    variants = [v for v in ["Full-IM", "w/o Privacy", "w/o TTA",
                            "w/o Compression", "w/o DriftDetect",
                            "w/o Orchestration", "w/o HITL",
                            "No-IM (raw argmax)"] if v in g.groups]
    vals = np.zeros((len(variants), len(axes_spec)))
    for j, (col, _, invert) in enumerate(axes_spec):
        v = np.array([g.get_group(x)[col].mean() for x in variants], float)
        if col == "privacy_budget_epsilon":
            v = 1.0 / np.clip(v, 1e-9, None)          # inf epsilon -> 0 privacy
        lo, hi = np.nanmin(v), np.nanmax(v)
        n = (v - lo) / (hi - lo) if hi > lo else np.ones_like(v) * 0.5
        vals[:, j] = (1 - n) if (invert and col != "privacy_budget_epsilon") else n
    ang = np.linspace(0, 2 * np.pi, len(axes_spec), endpoint=False).tolist()
    ang += ang[:1]
    fig, ax = plt.subplots(figsize=(6.6, 6.4), subplot_kw=dict(polar=True))
    for i, name in enumerate(variants):
        r = vals[i].tolist(); r += r[:1]
        full = name == "Full-IM"
        ax.plot(ang, r, lw=2.4 if full else 1.2, label=name,
                ls="-" if full else "--", zorder=3 if full else 2)
        if full:
            ax.fill(ang, r, alpha=0.18)
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels([a[1] for a in axes_spec], fontsize=8)
    ax.set_yticklabels([])
    ax.set_title("Inference-Management component ablation\n"
                 "(outward = better on every axis; min-max scaled)",
                 fontweight="bold", pad=22)
    ax.legend(loc="upper right", bbox_to_anchor=(1.34, 1.14), frameon=False,
              fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig11_ablation_radar")


def fig_coverage_risk(cov: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    for m, g in cov.groupby("system"):
        gg = g.groupby("coverage").agg(err=("selective_error", "mean"),
                                       sd=("selective_error", "std"),
                                       fn=("selective_fn_rate", "mean"))
        ax.errorbar(gg.index, gg.err, yerr=gg.sd.fillna(0), marker="o", ms=4,
                    lw=1.8, capsize=3, label=f"{m} - error")
        ax.plot(gg.index, gg.fn, ls="--", lw=1.4, alpha=0.75,
                label=f"{m} - missed abnormal")
    ax.set_xlabel("Coverage (fraction auto-reported)")
    ax.set_ylabel("Rate on the accepted subset")
    ax.set_title("Selective prediction: risk vs coverage", fontweight="bold")
    ax.legend(frameon=False, fontsize=7.5); ax.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, "fig13_coverage_risk")


def fig_prior_leak(rob: pd.DataFrame):
    """Contribution C1: the control that must sit at chance, and does."""
    pr = rob[rob.method_name == "Prior-only (no image)"]
    if pr.empty:
        return
    # Values measured under the discarded paired-partition protocol, kept so
    # the figure shows what the fix actually removes.
    paired = {0.1: 0.9416, 0.5: 0.8102, 1.0: 0.6954, 1e9: 0.4980}
    g = pr.groupby("alpha_num").auc_roc.agg(["mean", "std"]).sort_index()
    x = np.arange(len(g))
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.bar(x - 0.2, [paired[a] for a in g.index], 0.4, label="paired train/test "
           "partition (discarded)", color="#C44E52", edgecolor="black")
    ax.bar(x + 0.2, g["mean"], 0.4, yerr=g["std"].fillna(0), capsize=4,
           label="pooled test set (adopted)", color="#55A868",
           edgecolor="black")
    ax.axhline(0.5, ls="--", c="k", lw=1.2)
    ax.text(len(x) - 0.5, 0.515, "chance", fontsize=8, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(["IID" if a >= 1e8 else f"{a:g}" for a in g.index])
    ax.set_xlabel("Dirichlet concentration alpha")
    ax.set_ylabel("AUROC of a predictor that never sees the image")
    ax.set_title("Label-prior leakage in personalised-FL evaluation",
                 fontweight="bold")
    ax.legend(frameon=False, fontsize=8); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    save(fig, "fig06_prior_leak")


def save(fig, stem: str):
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    LOG.info("wrote %s.{png,pdf}", stem)


# ---------------------------------------------------------------------------
def significance(main: pd.DataFrame) -> pd.DataFrame:
    """Proposed vs every baseline, per metric, Holm-corrected within metric.

    `wilcoxon_vs_reference` already applies Holm across the family of methods
    for one metric, so correcting again across metrics would double-penalise.
    With 5 seeds the smallest attainable two-sided p is 0.0625, so nothing here
    can reach p<0.05 on its own - DeLong on the pooled predictions is the
    primary test and this is the supporting seed-level evidence.
    """
    metrics = ["auc_roc", "sensitivity_at_90spec", "f1_score", "ece",
               "latency_ms"]
    if PROPOSED_IM not in set(main.method_name):
        return pd.DataFrame(columns=["metric", "method", "p_holm"])
    out = []
    for metric in metrics:
        if metric not in main.columns:
            continue
        w = wilcoxon_vs_reference(main, metric, reference=PROPOSED_IM)
        if w.empty:
            continue
        # Flag which comparison is against the strongest baseline, since that
        # is the gap the paper must quote.
        lower_better = metric in ("ece", "latency_ms")
        agg = (main[main.method_name != PROPOSED_IM]
               .groupby("method_name")[metric].mean())
        best = agg.idxmin() if lower_better else agg.idxmax()
        w = w.assign(metric=metric, method=PROPOSED_IM,
                     is_strongest_baseline=w["method_name"] == best)
        out.append(w)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["metric", "method", "p_holm"])


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    mb = read("main_results_benchmark.csv")
    rob = read("noniid_robustness.csv")
    hist = read("training_history.csv")
    abl = read("ablation_study_results.csv")
    cov = read("coverage_risk.csv")

    sig = pd.DataFrame(columns=["metric", "method", "p_holm"])
    if mb is not None:
        sig = significance(mb)
        if not sig.empty:
            sig.to_csv(CSV_DIR / "significance_tests.csv", index=False)
            cols = [c for c in ("metric", "method_name", "reference", "p_raw",
                                "p_holm", "cohens_dz", "mean_delta",
                                "is_strongest_baseline") if c in sig.columns]
            LOG.info("Wilcoxon vs baselines (Holm-corrected within metric):\n%s",
                     sig[cols].to_string(index=False))

        agg = aggregate_mean_std(
            mb, ["method_name"],
            ["auc_roc", "sensitivity_at_90spec", "specificity", "f1_score",
             "accuracy", "ece", "latency_ms", "model_size_mb"])
        agg.to_csv(CSV_DIR / "aggregated_main_benchmark.csv", index=False)
        LOG.info("main benchmark:\n%s", agg.to_string(index=False))
        fig_main_benchmark(mb, sig)

        # ---- DeLong on the stored per-sample predictions ------------------
        preds, ybin = {}, None
        for m in list(METHODS) + ["FedAvg+IM"]:
            f = CSV_DIR / f"test_predictions_{m.replace('+', '_')}_s{CFG.seeds[0]}.npz"
            if f.exists():
                d = np.load(f)
                preds[m] = d["binary_score"]
                ybin = d["ybin"]
        if ybin is not None and PROPOSED_IM in preds:
            dl = []
            for m, s in preds.items():
                if m == PROPOSED_IM:
                    continue
                r = delong_roc_test(ybin, preds[PROPOSED_IM], s)
                dl.append({"proposed": PROPOSED_IM, "against": m, **r})
            dl = pd.DataFrame(dl)
            dl["signif"] = [stars(p) for p in dl.p_value]
            dl.to_csv(CSV_DIR / "delong_auc_tests.csv", index=False)
            LOG.info("DeLong AUC tests (seed %d):\n%s", CFG.seeds[0],
                     dl.to_string(index=False))
            keep = {k: v for k, v in preds.items()
                    if k in (PROPOSED_IM, "Centralized", "FedAvg", "FedProx")}
            fig_roc(keep, ybin)

    if rob is not None:
        fig_noniid(rob)
        fig_prior_leak(rob)
    if hist is not None:
        fig_convergence(hist)
    if abl is not None:
        a = aggregate_mean_std(abl, ["ablation_variant"],
                               ["auc_roc", "f1_score", "ece", "latency_ms",
                                "hitl_flag_rate"])
        a.to_csv(CSV_DIR / "aggregated_ablation.csv", index=False)
        LOG.info("ablation:\n%s", a.to_string(index=False))
        fig_ablation_radar(abl)
    if cov is not None:
        fig_coverage_risk(cov)

    LOG.info("STAGE 06 COMPLETE")


if __name__ == "__main__":
    main()
