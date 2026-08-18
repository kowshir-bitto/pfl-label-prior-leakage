"""
Metrics, calibration, and the statistical-significance protocol.

Contains
--------
* binary screening metrics (AUC, Sens@90%Spec, specificity, F1, accuracy)
* Expected Calibration Error over 10 equal-width confidence bins
* fast DeLong (Sun & Xu, 2014) for paired AUC comparison
* Wilcoxon signed-rank across seeds with Holm-Bonferroni family-wise control
* stratified bootstrap 95% CI bands for ROC curves
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             roc_curve)

from .config import CFG


# ---------------------------------------------------------------------------
# Screening metrics
# ---------------------------------------------------------------------------
def sensitivity_at_specificity(y_true: np.ndarray, score: np.ndarray,
                               spec_level: float = 0.90) -> tuple[float, float, float]:
    """Sensitivity at the operating point where specificity first reaches
    `spec_level`.  Returns (sensitivity, achieved_specificity, threshold).

    This is the clinically meaningful operating point for a screening triage
    tool: the specificity floor is fixed by how many false alarms an endoscopy
    unit can absorb, and sensitivity is what we then maximise.
    """
    fpr, tpr, thr = roc_curve(y_true, score)
    spec = 1.0 - fpr
    ok = np.where(spec >= spec_level)[0]
    if len(ok) == 0:
        return 0.0, float(spec.max()), float(thr[int(np.argmax(spec))])
    i = ok[int(np.argmax(tpr[ok]))]
    return float(tpr[i]), float(spec[i]), float(thr[i])


def binary_screening_metrics(y_true: np.ndarray, score: np.ndarray,
                             spec_level: float | None = None) -> dict:
    spec_level = spec_level or CFG.eval.spec_operating_point
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score, dtype=float)
    if len(np.unique(y_true)) < 2:
        return dict(auc_roc=np.nan, sensitivity_at_90spec=np.nan,
                    specificity=np.nan, f1_score=np.nan, accuracy=np.nan,
                    threshold=0.5)
    auc = float(roc_auc_score(y_true, score))
    sens, spec, thr = sensitivity_at_specificity(y_true, score, spec_level)
    pred = (score >= thr).astype(int)
    return dict(
        auc_roc=auc,
        sensitivity_at_90spec=sens,
        specificity=spec,
        f1_score=float(f1_score(y_true, pred, zero_division=0)),
        accuracy=float(accuracy_score(y_true, pred)),
        threshold=float(thr),
    )


def multiclass_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict:
    pred = probs.argmax(1)
    out = dict(
        accuracy_6c=float(accuracy_score(y_true, pred)),
        macro_f1_6c=float(f1_score(y_true, pred, average="macro",
                                   zero_division=0)),
        weighted_f1_6c=float(f1_score(y_true, pred, average="weighted",
                                      zero_division=0)),
    )
    try:
        out["macro_auc_6c"] = float(roc_auc_score(
            y_true, probs, multi_class="ovr", average="macro"))
    except ValueError:
        out["macro_auc_6c"] = np.nan
    return out


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def expected_calibration_error(probs: np.ndarray, labels: np.ndarray,
                               n_bins: int | None = None) -> float:
    """ECE over equal-width bins of the top-1 confidence."""
    n_bins = n_bins or CFG.eval.ece_bins
    conf = probs.max(1)
    pred = probs.argmax(1)
    correct = (pred == labels).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, n = 0.0, len(labels)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        ece += (m.sum() / n) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def reliability_bins(probs: np.ndarray, labels: np.ndarray,
                     n_bins: int | None = None) -> pd.DataFrame:
    n_bins = n_bins or CFG.eval.ece_bins
    conf = probs.max(1)
    correct = (probs.argmax(1) == labels).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        rows.append({
            "bin_lo": lo, "bin_hi": hi, "bin_mid": (lo + hi) / 2,
            "count": int(m.sum()),
            "avg_conf": float(conf[m].mean()) if m.sum() else np.nan,
            "accuracy": float(correct[m].mean()) if m.sum() else np.nan,
        })
    return pd.DataFrame(rows)


def brier_score(binary_score: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.asarray(binary_score) - np.asarray(y)) ** 2))


# ---------------------------------------------------------------------------
# DeLong's test (fast algorithm, Sun & Xu 2014)
# ---------------------------------------------------------------------------
def _midrank(x: np.ndarray) -> np.ndarray:
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(N, dtype=float)
    out[J] = T
    return out


def _fast_delong(preds_sorted: np.ndarray, m: int):
    """`preds_sorted` is (k, N) with the m positives first."""
    k, N = preds_sorted.shape
    n = N - m
    pos, neg = preds_sorted[:, :m], preds_sorted[:, m:]
    tx = np.empty((k, m)); ty = np.empty((k, n)); tz = np.empty((k, N))
    for r in range(k):
        tx[r] = _midrank(pos[r])
        ty[r] = _midrank(neg[r])
        tz[r] = _midrank(preds_sorted[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01) if k > 1 else np.array([[np.var(v01, ddof=1)]])
    sy = np.cov(v10) if k > 1 else np.array([[np.var(v10, ddof=1)]])
    cov = np.atleast_2d(sx) / m + np.atleast_2d(sy) / n
    return aucs, cov


def delong_roc_test(y_true: np.ndarray, score_a: np.ndarray,
                    score_b: np.ndarray) -> dict:
    """Paired, non-parametric comparison of two correlated ROC AUCs.

    Both scores must come from the SAME samples in the SAME order - which the
    paired Dirichlet test-shard design guarantees.
    """
    y = np.asarray(y_true).astype(int)
    order = np.argsort(-y, kind="stable")     # positives first
    y_s = y[order]
    m = int(y_s.sum())
    preds = np.vstack([np.asarray(score_a, float)[order],
                       np.asarray(score_b, float)[order]])
    aucs, cov = _fast_delong(preds, m)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        z, p = 0.0, 1.0
    else:
        z = (aucs[0] - aucs[1]) / np.sqrt(var)
        p = float(2 * sps.norm.sf(abs(z)))
    return dict(auc_a=float(aucs[0]), auc_b=float(aucs[1]),
                auc_diff=float(aucs[0] - aucs[1]),
                z=float(z), p_value=p, var=float(var))


def delong_ci(y_true: np.ndarray, score: np.ndarray,
              conf: float = 0.95) -> tuple[float, float, float]:
    y = np.asarray(y_true).astype(int)
    order = np.argsort(-y, kind="stable")
    m = int(y[order].sum())
    aucs, cov = _fast_delong(np.vstack([np.asarray(score, float)[order]]), m)
    se = float(np.sqrt(max(cov[0, 0], 0)))
    z = sps.norm.ppf(1 - (1 - conf) / 2)
    a = float(aucs[0])
    return a, max(0.0, a - z * se), min(1.0, a + z * se)


# ---------------------------------------------------------------------------
# Multi-seed aggregation and family-wise correction
# ---------------------------------------------------------------------------
def holm_bonferroni(pvals: dict, alpha: float | None = None) -> pd.DataFrame:
    """Step-down Holm correction over a family of comparisons."""
    alpha = alpha or CFG.eval.alpha_significance
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    k = len(items)
    rows, prev = [], 0.0
    for i, (name, p) in enumerate(items):
        adj = min(1.0, max(prev, (k - i) * p))
        prev = adj
        rows.append({"comparison": name, "p_raw": p, "p_holm": adj,
                     "significant": adj < alpha})
    return pd.DataFrame(rows)


def wilcoxon_vs_reference(df: pd.DataFrame, metric: str,
                          method_col: str = "method_name",
                          seed_col: str = "seed",
                          reference: str | None = None) -> pd.DataFrame:
    """Paired Wilcoxon signed-rank of every method against the reference.

    Pairing is by seed: with 5 seeds the smallest attainable two-sided exact
    p-value is 0.0625, so we report the raw statistic alongside the Holm-
    adjusted p and never over-claim significance from the seed test alone -
    DeLong on the pooled test predictions is the primary inferential test.
    """
    from .config import PROPOSED_METHOD
    reference = reference or PROPOSED_METHOD
    piv = df.pivot_table(index=seed_col, columns=method_col, values=metric)
    if reference not in piv.columns:
        return pd.DataFrame()
    raw, effects = {}, {}
    for m in piv.columns:
        if m == reference:
            continue
        a, b = piv[reference].to_numpy(), piv[m].to_numpy()
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 3 or np.allclose(a[ok], b[ok]):
            raw[m], effects[m] = 1.0, 0.0
            continue
        try:
            stat = sps.wilcoxon(a[ok], b[ok], zero_method="wilcox",
                                alternative="two-sided")
            raw[m] = float(stat.pvalue)
        except ValueError:
            raw[m] = 1.0
        diff = a[ok] - b[ok]
        effects[m] = float(diff.mean() / (diff.std(ddof=1) + 1e-12))
    out = holm_bonferroni(raw)
    out = out.rename(columns={"comparison": method_col})
    out["reference"] = reference
    out["metric"] = metric
    out["cohens_dz"] = out[method_col].map(effects)
    out["mean_delta"] = out[method_col].map(
        {m: float(np.nanmean(piv[reference] - piv[m])) for m in raw})
    return out


def mean_std_ci(x, conf: float = 0.95) -> dict:
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    n = len(a)
    if n == 0:
        return dict(mean=np.nan, std=np.nan, ci_lo=np.nan, ci_hi=np.nan, n=0)
    mu, sd = float(a.mean()), float(a.std(ddof=1)) if n > 1 else 0.0
    if n > 1:
        h = sps.t.ppf(1 - (1 - conf) / 2, n - 1) * sd / np.sqrt(n)
    else:
        h = 0.0
    return dict(mean=mu, std=sd, ci_lo=mu - h, ci_hi=mu + h, n=n)


def aggregate_mean_std(df: pd.DataFrame, group_cols: list[str],
                       metrics: list[str]) -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        rec = dict(zip(group_cols, keys))
        for m in metrics:
            if m not in g.columns:
                continue
            s = mean_std_ci(g[m].to_numpy())
            rec[f"{m}_mean"] = s["mean"]
            rec[f"{m}_std"] = s["std"]
            rec[f"{m}_ci_lo"] = s["ci_lo"]
            rec[f"{m}_ci_hi"] = s["ci_hi"]
            rec[f"{m}_str"] = (f"{s['mean']:.4f} ± {s['std']:.4f}"
                               if np.isfinite(s["mean"]) else "n/a")
        rec["n_seeds"] = len(g)
        rows.append(rec)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Bootstrap ROC band
# ---------------------------------------------------------------------------
def bootstrap_roc_band(y_true: np.ndarray, score: np.ndarray,
                       n_boot: int | None = None, seed: int = 0,
                       grid: np.ndarray | None = None):
    """Stratified bootstrap 95% band for a ROC curve on a fixed FPR grid."""
    n_boot = n_boot or CFG.eval.n_bootstrap
    rng = np.random.RandomState(seed)
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score, float)
    grid = np.linspace(0, 1, 101) if grid is None else grid
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    curves = np.empty((n_boot, len(grid)))
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos), replace=True),
                              rng.choice(neg, len(neg), replace=True)])
        fpr, tpr, _ = roc_curve(y[idx], s[idx])
        curves[b] = np.interp(grid, fpr, tpr)
    fpr0, tpr0, _ = roc_curve(y, s)
    return (grid, np.interp(grid, fpr0, tpr0),
            np.percentile(curves, 2.5, axis=0),
            np.percentile(curves, 97.5, axis=0))
