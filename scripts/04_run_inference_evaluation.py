"""
04 - STEP 1: Inference-management evaluation.

This is where the paper's claim is tested.  Every trained model from stage 03
is evaluated twice:

  * baselines           -> vanilla inference (single view, argmax, no gating).
                           This is what a deployed FL model normally does.
  * proposed (FedGIM)   -> wrapped in the full Inference Management layer.
  * FedAvg + IM         -> the SAME IM layer bolted onto a baseline backbone,
                           which isolates the IM contribution from the FedGIM
                           training recipe and demonstrates model-agnosticism.

It additionally runs
  * the 8-variant IM ablation (six single-component knock-outs + full + none),
  * a differential-privacy epsilon sweep,
  * a covariate-drift severity sweep with detector ROC,
  * a selective-prediction (HITL) coverage-risk sweep,
  * per-lesion subgroup error analysis.

Outputs
-------
    outputs/csv/main_results_benchmark.csv
    outputs/csv/ablation_study_results.csv
    outputs/csv/subgroup_lesion_analysis.csv
    outputs/csv/privacy_utility_sweep.csv
    outputs/csv/drift_detection.csv
    outputs/csv/coverage_risk.csv
    outputs/csv/test_predictions_{method}_s{seed}.npz
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from src.config import (ABNORMAL_CLASSES, CACHE_DIR, CFG, CKPT_DIR,
                        CLASS_TO_IDX, CLASSES, CSV_DIR, IM_TRANSFER_BASE,
                        LESION_STRATA, METHODS, MODALITY, N_CLASSES,
                        PROPOSED_METHOD, SCHEMA_ABLATION, SCHEMA_MAIN,
                        SCHEMA_SUBGROUP)
from src.inference_manager import (InferenceManager, inject_covariate_drift,
                                   module_size_mb)
from src.models import MultiEncoder, make_head
from src.stats_utils import (binary_screening_metrics,
                             expected_calibration_error, multiclass_metrics)
from src.utils import configure_torch, get_logger, set_seed

LOG = get_logger("04_inference")
ABN_IDX = [CLASS_TO_IDX[c] for c in ABNORMAL_CLASSES]
IM = CFG.im
MAIN_A = "0.5"

ALL_OFF = {k: False for k in ("privacy", "tta", "compression", "drift",
                              "orchestration", "hitl")}
ALL_ON = {k: True for k in ALL_OFF}

ABLATION_MAP = {
    "Full-IM": ALL_ON,
    "w/o Privacy": {**ALL_ON, "privacy": False},
    "w/o TTA": {**ALL_ON, "tta": False},
    "w/o Compression": {**ALL_ON, "compression": False},
    "w/o DriftDetect": {**ALL_ON, "drift": False},
    "w/o Orchestration": {**ALL_ON, "orchestration": False},
    "w/o HITL": {**ALL_ON, "hitl": False},
    "No-IM (raw argmax)": ALL_OFF,
}


# ---------------------------------------------------------------------------
def measure_encoder_cost() -> tuple[float, float]:
    """Single-frame encoder latency and on-disk size on THIS CPU.

    Reported honestly as an edge-deployment measurement (Intel i7-1355U,
    CPU-only); it is identical for every method, so all latency differences in
    the benchmark come from the head and the IM layer.
    """
    enc = MultiEncoder(CFG.model.backbones, pretrained=True).eval()
    x = torch.randn(1, 3, CFG.preproc.img_size, CFG.preproc.img_size)
    with torch.no_grad():
        for _ in range(CFG.eval.latency_warmup):
            enc(x)
        t0 = time.perf_counter()
        for _ in range(CFG.eval.latency_repeats):
            enc(x)
        ms = (time.perf_counter() - t0) / CFG.eval.latency_repeats * 1000
    mb = module_size_mb(enc)
    LOG.info("encoder: %.2f ms/frame, %.2f MB (frozen, shared by all methods)",
             ms, mb)
    del enc
    return ms, mb


def load_ckpt(method: str, seed: int):
    p = CKPT_DIR / f"{method}_s{seed}_a{MAIN_A}.pt"
    if not p.exists():
        return None
    return torch.load(p, map_location="cpu", weights_only=False)


def heads_from_ckpt(ck) -> tuple[list, object | None, object | None]:
    """Rebuild the heads, the feature-space router, and the global head.

    Returns `(heads, router, global_head)`.  Routing comes from the saved
    training-feature centroids rather than ground-truth test shards, so a
    personalised head can no longer be handed the samples that match its own
    label prior.
    """
    from src.fl_algorithms import ClientRouter, personal_head
    in_dim = ck["in_dim"]
    if not ck["is_personalized"]:
        h = make_head(in_dim, N_CLASSES)
        h.load_state_dict(ck["global_state"])
        h.eval()
        return [h], None, None

    heads = [personal_head(ck["method"], in_dim, ck["global_state"], cs)
             for cs in ck["client_states"]]

    router = ClientRouter.from_centroids(ck["router_centroids"])
    # Local-only never forms a server model, so it has no global fallback.
    g = None
    if ck["method"] != "Local-only":
        g = make_head(in_dim, N_CLASSES)
        g.load_state_dict(ck["global_state"])
        g.eval()
    return heads, router, g


def evaluate(out, y6: np.ndarray, ybin: np.ndarray, selective: bool = True
             ) -> dict:
    """Score an IMOutput.

    When HITL gating is active the operating metrics are computed on the
    ACCEPTED subset - that is the population the system actually auto-reports
    on, and reporting anything else would hide the benefit of abstention.  The
    referred fraction is always disclosed as `hitl_flag_rate`.
    """
    mask = out.accepted if (selective and out.accepted.any()) else np.ones(
        len(y6), dtype=bool)
    m = binary_screening_metrics(ybin[mask], out.binary_score[mask])
    m.update(multiclass_metrics(y6[mask], out.probs[mask]))
    m["ece"] = expected_calibration_error(out.probs[mask], y6[mask])
    m["hitl_flag_rate"] = out.hitl_flag_rate
    m["escalation_rate"] = out.escalation_rate
    m["coverage"] = float(mask.mean())
    m["latency_ms"] = out.latency_ms
    m["throughput_fps"] = out.throughput_fps
    m["model_size_mb"] = out.model_size_mb
    m["drift_detected"] = bool(out.drift_detected)
    m["drift_mmd"] = out.drift_mmd
    m["privacy_budget_epsilon"] = out.epsilon
    return m


# ---------------------------------------------------------------------------
def main() -> None:
    configure_torch()
    test_Z = torch.from_numpy(np.load(CACHE_DIR / "feat_test.npy"))
    val_Z = np.load(CACHE_DIR / "feat_val.npy")[:, 0, :]
    meta = pd.read_csv(CACHE_DIR / "meta_test.csv")
    y6 = meta.class_idx.to_numpy()
    ybin = meta.binary_label.to_numpy()
    LOG.info("test set: %d images | abnormal=%d normal=%d",
             len(y6), int(ybin.sum()), int((1 - ybin).sum()))

    enc_ms, enc_mb = measure_encoder_cost()

    main_rows, abl_rows, sub_rows = [], [], []
    priv_rows, drift_rows, cov_rows = [], [], []

    # ---- benchmark: every method, vanilla vs IM-wrapped ------------------
    bench = list(METHODS) + [f"{IM_TRANSFER_BASE}+IM"]
    for seed in CFG.seeds:
        set_seed(seed)
        for name in bench:
            base = IM_TRANSFER_BASE if name.endswith("+IM") else name
            ck = load_ckpt(base, seed)
            if ck is None:
                LOG.warning("missing checkpoint %s s%d", base, seed)
                continue
            heads, router, g_head = heads_from_ckpt(ck)

            use_im = (name == PROPOSED_METHOD) or name.endswith("+IM")
            enabled = ALL_ON if use_im else ALL_OFF

            mgr = InferenceManager(heads, router, val_Z, global_head=g_head,
                                   cfg=IM, enabled=enabled, seed=seed,
                                   encoder_latency_ms=enc_ms,
                                   encoder_size_mb=enc_mb)
            out = mgr.run(test_Z)
            # The headline columns are always FULL-COVERAGE.  Scoring an
            # abstaining system on the cases it chose to keep, against
            # baselines scored on every case, would credit the IM layer for
            # the difficulty of the cases it declined rather than for its
            # predictions.  The selective numbers are reported alongside, and
            # the coverage-risk sweep below is where abstention is evaluated
            # on its own terms.
            m = evaluate(out, y6, ybin, selective=False)
            ms = evaluate(out, y6, ybin, selective=True) if use_im else m

            main_rows.append({
                "seed": seed, "method_name": name, "modality": MODALITY,
                "auc_roc": m["auc_roc"],
                "sensitivity_at_90spec": m["sensitivity_at_90spec"],
                "specificity": m["specificity"], "f1_score": m["f1_score"],
                "accuracy": m["accuracy"], "latency_ms": m["latency_ms"],
                "throughput_fps": m["throughput_fps"],
                "model_size_mb": m["model_size_mb"], "ece": m["ece"],
                # extended columns (kept beyond the required schema)
                "macro_f1_6c": m["macro_f1_6c"], "accuracy_6c": m["accuracy_6c"],
                "macro_auc_6c": m["macro_auc_6c"],
                # matched-coverage view of the same run
                "coverage": ms["coverage"],
                "auc_roc_selective": ms["auc_roc"],
                "sens90_selective": ms["sensitivity_at_90spec"],
                "ece_selective": ms["ece"],
                "hitl_flag_rate": m["hitl_flag_rate"],
                "escalation_rate": m["escalation_rate"],
                "comm_mb_total": ck["comm_mb_total"],
                "im_enabled": use_im,
            })
            np.savez_compressed(
                CSV_DIR / f"test_predictions_{name.replace('+', '_')}_s{seed}.npz",
                probs=out.probs, binary_score=out.binary_score,
                accepted=out.accepted, y6=y6, ybin=ybin)

            LOG.info("seed %-4d %-12s AUC=%.4f Sens@90=%.4f ECE=%.4f "
                     "lat=%.1fms flag=%.1f%%", seed, name, m["auc_roc"],
                     m["sensitivity_at_90spec"], m["ece"], m["latency_ms"],
                     100 * m["hitl_flag_rate"])

            # ---- subgroup analysis (proposed + key baselines) -------------
            if name in (PROPOSED_METHOD, "FedAvg", "Centralized",
                        f"{IM_TRANSFER_BASE}+IM"):
                thr = binary_screening_metrics(
                    ybin, out.binary_score)["threshold"]
                pred_bin = (out.binary_score >= thr).astype(int)
                norm_mask = ybin == 0
                for stratum in LESION_STRATA:
                    if stratum == "Normal":
                        sel = norm_mask
                        sub_rows.append({
                            "seed": seed, "method": name, "modality": MODALITY,
                            "lesion_stage": "Normal",
                            "total_samples": int(sel.sum()),
                            "false_negatives": 0,
                            "false_positives": int(pred_bin[sel].sum()),
                            "auc_roc": np.nan,
                            "sensitivity": np.nan,
                            "specificity": float(1 - pred_bin[sel].mean()),
                        })
                        continue
                    sel = y6 == CLASS_TO_IDX[stratum]
                    if sel.sum() == 0:
                        continue
                    pair = sel | norm_mask
                    try:
                        from sklearn.metrics import roc_auc_score
                        auc = float(roc_auc_score(ybin[pair],
                                                  out.binary_score[pair]))
                    except ValueError:
                        auc = np.nan
                    sub_rows.append({
                        "seed": seed, "method": name, "modality": MODALITY,
                        "lesion_stage": stratum,
                        "total_samples": int(sel.sum()),
                        "false_negatives": int((pred_bin[sel] == 0).sum()),
                        "false_positives": 0,
                        "auc_roc": auc,
                        "sensitivity": float(pred_bin[sel].mean()),
                        "specificity": np.nan,
                    })

    # ---- IM ablation on the proposed backbone ----------------------------
    LOG.info("--- IM ablation (%d variants x %d seeds) ---",
             len(ABLATION_MAP), len(CFG.seeds))
    for seed in CFG.seeds:
        set_seed(seed)
        ck = load_ckpt(PROPOSED_METHOD, seed)
        if ck is None:
            continue
        heads, router, g_head = heads_from_ckpt(ck)
        for variant, enabled in ABLATION_MAP.items():
            mgr = InferenceManager(heads, router, val_Z, global_head=g_head,
                                   cfg=IM, enabled=enabled, seed=seed,
                                   encoder_latency_ms=enc_ms,
                                   encoder_size_mb=enc_mb)
            out = mgr.run(test_Z)
            # Full coverage for the headline ablation columns.  Knocking out
            # HITL changes how many cases are scored, so a selective metric
            # would compare variants on different populations and the
            # "w/o HITL" row would look worse purely for answering harder
            # questions.  Selective values are carried alongside.
            m = evaluate(out, y6, ybin, selective=False)
            ms = (evaluate(out, y6, ybin, selective=True)
                  if enabled["hitl"] else m)
            abl_rows.append({
                "seed": seed, "ablation_variant": variant,
                "auc_roc": m["auc_roc"], "f1_score": m["f1_score"],
                "latency_ms": m["latency_ms"],
                "hitl_flag_rate": m["hitl_flag_rate"], "ece": m["ece"],
                "privacy_budget_epsilon": m["privacy_budget_epsilon"],
                # extended
                "sensitivity_at_90spec": m["sensitivity_at_90spec"],
                "specificity": m["specificity"],
                "model_size_mb": m["model_size_mb"],
                "throughput_fps": m["throughput_fps"],
                "escalation_rate": m["escalation_rate"],
                "coverage": ms["coverage"],
                "auc_roc_selective": ms["auc_roc"],
                "ece_selective": ms["ece"],
            })
        LOG.info("  seed %d ablation done", seed)

    # ---- privacy / utility trade-off -------------------------------------
    LOG.info("--- privacy sweep ---")
    for seed in CFG.seeds:
        ck = load_ckpt(PROPOSED_METHOD, seed)
        if ck is None:
            continue
        heads, router, g_head = heads_from_ckpt(ck)
        for eps in IM.dp_epsilon_sweep:
            mgr = InferenceManager(heads, router, val_Z, global_head=g_head,
                                   cfg=IM, enabled=ALL_ON, seed=seed,
                                   encoder_latency_ms=enc_ms,
                                   encoder_size_mb=enc_mb)
            out = mgr.run(test_Z, epsilon=eps)
            m = evaluate(out, y6, ybin)
            priv_rows.append({
                "seed": seed, "epsilon": ("inf" if np.isinf(eps) else eps),
                "epsilon_num": (1e9 if np.isinf(eps) else float(eps)),
                "auc_roc": m["auc_roc"],
                "sensitivity_at_90spec": m["sensitivity_at_90spec"],
                "ece": m["ece"], "f1_score": m["f1_score"],
            })

    # ---- drift injection + detector behaviour ----------------------------
    LOG.info("--- drift sweep ---")
    for seed in CFG.seeds:
        rng = np.random.RandomState(seed)
        ck_p = load_ckpt(PROPOSED_METHOD, seed)
        ck_b = load_ckpt("FedAvg", seed)
        if ck_p is None or ck_b is None:
            continue
        for sev in IM.drift_severities:
            Zd = torch.from_numpy(
                inject_covariate_drift(test_Z.numpy(), sev, rng).astype(
                    np.float32))
            for tag, ck, enabled in (("FedGIM+IM", ck_p, ALL_ON),
                                     ("FedAvg (no IM)", ck_b, ALL_OFF)):
                heads, router, g_head = heads_from_ckpt(ck)
                mgr = InferenceManager(heads, router, val_Z,
                                       global_head=g_head, cfg=IM,
                                       enabled=enabled, seed=seed,
                                       encoder_latency_ms=enc_ms,
                                       encoder_size_mb=enc_mb)
                out = mgr.run(Zd)
                m = evaluate(out, y6, ybin, selective=enabled["hitl"])
                drift_rows.append({
                    "seed": seed, "severity": sev, "system": tag,
                    "auc_roc": m["auc_roc"],
                    "sensitivity_at_90spec": m["sensitivity_at_90spec"],
                    "ece": m["ece"],
                    "drift_detected": m["drift_detected"],
                    "drift_mmd": m["drift_mmd"],
                    "hitl_flag_rate": m["hitl_flag_rate"],
                })

    # ---- selective prediction: risk vs coverage --------------------------
    LOG.info("--- coverage-risk sweep ---")
    from src.inference_manager import hitl_scores
    for seed in CFG.seeds:
        for tag, base, enabled in (("FedGIM+IM", PROPOSED_METHOD, ALL_ON),
                                   ("FedAvg (no IM)", "FedAvg", ALL_OFF)):
            ck = load_ckpt(base, seed)
            if ck is None:
                continue
            heads, router, g_head = heads_from_ckpt(ck)
            mgr = InferenceManager(heads, router, val_Z, global_head=g_head,
                                   cfg=IM,
                                   enabled={**enabled, "hitl": False},
                                   seed=seed, encoder_latency_ms=enc_ms,
                                   encoder_size_mb=enc_mb)
            out = mgr.run(test_Z)
            ent, margin = hitl_scores(out.probs)
            unc = ent + (1 - margin)
            order = np.argsort(unc, kind="stable")
            thr = binary_screening_metrics(ybin, out.binary_score)["threshold"]
            pred = (out.binary_score >= thr).astype(int)
            for cov in np.arange(0.50, 1.001, 0.05):
                k = max(10, int(cov * len(order)))
                keep = order[:k]
                cov_rows.append({
                    "seed": seed, "system": tag, "coverage": float(cov),
                    "selective_error": float((pred[keep] != ybin[keep]).mean()),
                    "selective_fn_rate": float(
                        ((pred[keep] == 0) & (ybin[keep] == 1)).sum()
                        / max((ybin[keep] == 1).sum(), 1)),
                })

    # ---- write ------------------------------------------------------------
    main_df = pd.DataFrame(main_rows)
    abl_df = pd.DataFrame(abl_rows)
    sub_df = pd.DataFrame(sub_rows)
    main_df[SCHEMA_MAIN + [c for c in main_df.columns if c not in SCHEMA_MAIN]] \
        .to_csv(CSV_DIR / "main_results_benchmark.csv", index=False)
    abl_df[SCHEMA_ABLATION + [c for c in abl_df.columns
                              if c not in SCHEMA_ABLATION]] \
        .to_csv(CSV_DIR / "ablation_study_results.csv", index=False)
    sub_df[SCHEMA_SUBGROUP + [c for c in sub_df.columns
                              if c not in SCHEMA_SUBGROUP]] \
        .to_csv(CSV_DIR / "subgroup_lesion_analysis.csv", index=False)
    pd.DataFrame(priv_rows).to_csv(CSV_DIR / "privacy_utility_sweep.csv",
                                   index=False)
    pd.DataFrame(drift_rows).to_csv(CSV_DIR / "drift_detection.csv",
                                    index=False)
    pd.DataFrame(cov_rows).to_csv(CSV_DIR / "coverage_risk.csv", index=False)

    LOG.info("=" * 78)
    LOG.info("MAIN BENCHMARK (mean over %d seeds)", len(CFG.seeds))
    agg = (main_df.groupby("method_name")
           [["auc_roc", "sensitivity_at_90spec", "f1_score", "ece",
             "latency_ms", "model_size_mb"]].agg(["mean", "std"]))
    LOG.info("\n%s", agg.round(4).to_string())
    LOG.info("STAGE 04 COMPLETE")


if __name__ == "__main__":
    main()
