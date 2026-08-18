"""
03 - Federated training across all methods, seeds and non-IID severities.

Grid
----
    methods : Local-only, Centralized, FedAvg, FedProx, FedPer, pFedMe, FedGIM
    seeds   : 42, 123, 456, 789, 2024
    alphas  : 0.1, 0.5, 1.0, inf     (Dirichlet label skew across 5 clients)

For the headline setting (alpha = 0.5) the full model state is checkpointed so
that stage 04 (inference management) and stage 05 (XAI) can reload it.  For the
remaining alphas only test predictions and metrics are retained - that is all
the robustness figure needs, and it keeps the checkpoint directory small.

Outputs
-------
    outputs/checkpoints/{method}_s{seed}_a{alpha}.pt
    outputs/csv/training_history.csv
    outputs/csv/noniid_robustness.csv
    outputs/csv/client_partitions.csv
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from src.config import (ABNORMAL_CLASSES, CACHE_DIR, CFG, CKPT_DIR,
                        CLASS_TO_IDX, CLASSES, CSV_DIR, METHODS, N_CLASSES)
from src.data_utils import dirichlet_partition
from src.fl_algorithms import (ClientData, ClientRouter, predict_test,
                               train_federated)
from src.stats_utils import binary_screening_metrics
from src.utils import configure_torch, get_logger, set_seed

LOG = get_logger("03_fedtrain")
ABN_IDX = [CLASS_TO_IDX[c] for c in ABNORMAL_CLASSES]


def alpha_tag(a: float) -> str:
    return "inf" if np.isinf(a) else f"{a:g}"


def load_bank():
    feats, metas = {}, {}
    for split in ("train", "val", "test"):
        feats[split] = torch.from_numpy(
            np.load(CACHE_DIR / f"feat_{split}.npy"))
        metas[split] = pd.read_csv(CACHE_DIR / f"meta_{split}.csv")
    with open(CACHE_DIR / "feature_bank_info.json") as f:
        info = json.load(f)
    return feats, metas, info


def build_clients(train_Z, train_meta, shards) -> list[ClientData]:
    y = torch.from_numpy(train_meta.class_idx.to_numpy()).long()
    yb = torch.from_numpy(train_meta.binary_label.to_numpy()).long()
    return [ClientData(cid=i, Z=train_Z[idx], y=y[idx], ybin=yb[idx])
            for i, idx in enumerate(shards)]


def main() -> None:
    configure_torch()
    feats, metas, info = load_bank()
    in_dim = int(info["dim"])
    LOG.info("feature bank: dim=%d counts=%s", in_dim, info["counts"])

    train_y = metas["train"].class_idx.to_numpy()
    test_y = metas["test"].class_idx.to_numpy()
    test_ybin = metas["test"].binary_label.to_numpy()
    val_Z, val_ybin = feats["val"], torch.from_numpy(
        metas["val"].binary_label.to_numpy())

    hist_rows, robust_rows, part_rows = [], [], []
    t_start = time.time()
    total_runs = len(METHODS) * len(CFG.seeds) * len(CFG.fed.alphas)
    run_i = 0

    for alpha in CFG.fed.alphas:
        atag = alpha_tag(alpha)
        is_main = (not np.isinf(alpha)) and abs(alpha - CFG.fed.main_alpha) < 1e-9

        for seed in CFG.seeds:
            set_seed(seed)
            # Non-IID is a *training-side* phenomenon: only the training set is
            # sharded across institutions.  The held-out test set stays pooled
            # and keeps the global class distribution, so it stands in for the
            # screening population every site actually faces.
            tr_shards = dirichlet_partition(train_y, CFG.fed.n_clients,
                                            alpha, seed)
            clients = build_clients(feats["train"], metas["train"], tr_shards)
            router = ClientRouter(clients, view=0)

            for cid, idx in enumerate(tr_shards):
                cnt = np.bincount(train_y[idx], minlength=N_CLASSES)
                part_rows.append({
                    "alpha": atag, "seed": seed, "client": cid,
                    "n_train": len(idx),
                    "pct_abnormal": float(100 * np.isin(train_y[idx],
                                                        ABN_IDX).mean()),
                    **{f"n_{CLASSES[c]}": int(cnt[c]) for c in range(N_CLASSES)},
                })

            # Integrity control: a predictor that never sees an image and only
            # knows each client's training label prior.  On the pooled test set
            # this must sit at chance; it is reported in the paper so readers
            # can verify the partition leaks nothing into the evaluation.
            # Every pooled test sample gets the same client-agnostic prior, so
            # this lands at chance by construction - which is exactly the
            # property the old paired partition violated (AUC 0.81 at a=0.5).
            cnt_all = np.bincount(train_y, minlength=N_CLASSES).astype(float)
            prior_score = np.full(len(test_y),
                                  cnt_all[ABN_IDX].sum() / cnt_all.sum())
            prior_score += np.random.RandomState(seed).normal(0, 1e-9,
                                                              len(prior_score))
            pm = binary_screening_metrics(test_ybin, prior_score)
            robust_rows.append({
                "alpha": atag,
                "alpha_num": (1e9 if np.isinf(alpha) else float(alpha)),
                "seed": seed, "method_name": "Prior-only (no image)",
                **{k: v for k, v in pm.items() if k != "threshold"},
                "comm_mb_total": 0.0, "train_sec": 0.0,
            })

            for method in METHODS:
                run_i += 1
                t0 = time.time()
                res = train_federated(
                    method, clients, in_dim, seed, alpha,
                    val_Z=val_Z, val_ybin=val_ybin,
                    rounds=CFG.fed.rounds, logger=None)
                dt = time.time() - t0

                logits = predict_test(res, feats["test"], router=router,
                                      view=0, mode="route")
                probs = np.exp(logits - logits.max(1, keepdims=True))
                probs /= probs.sum(1, keepdims=True)
                score = probs[:, ABN_IDX].sum(1)
                m = binary_screening_metrics(test_ybin, score)

                h = res.history.copy()
                h["method_name"] = method
                h["seed"] = seed
                h["alpha"] = atag
                hist_rows.append(h)

                robust_rows.append({
                    "alpha": atag, "alpha_num": (1e9 if np.isinf(alpha)
                                                 else float(alpha)),
                    "seed": seed, "method_name": method,
                    **{k: v for k, v in m.items() if k != "threshold"},
                    "comm_mb_total": res.comm_mb_total,
                    "train_sec": dt,
                })

                if is_main:
                    torch.save({
                        "method": method, "seed": seed, "alpha": alpha,
                        "is_personalized": res.is_personalized,
                        "global_state": res.global_state,
                        "client_states": res.client_states,
                        "in_dim": in_dim,
                        "train_shards": [s.tolist() for s in tr_shards],
                        "router_centroids": router.centroids.clone(),
                        "comm_mb_total": res.comm_mb_total,
                    }, CKPT_DIR / f"{method}_s{seed}_a{atag}.pt")

                eta = (time.time() - t_start) / run_i * (total_runs - run_i)
                LOG.info("[%3d/%3d] a=%-4s seed=%-4d %-12s AUC=%.4f "
                         "Sens@90=%.4f  %.1fs  ETA %.1f min",
                         run_i, total_runs, atag, seed, method,
                         m["auc_roc"], m["sensitivity_at_90spec"], dt, eta / 60)

    pd.concat(hist_rows, ignore_index=True).to_csv(
        CSV_DIR / "training_history.csv", index=False)
    pd.DataFrame(robust_rows).to_csv(
        CSV_DIR / "noniid_robustness.csv", index=False)
    pd.DataFrame(part_rows).to_csv(
        CSV_DIR / "client_partitions.csv", index=False)

    rb = pd.DataFrame(robust_rows)
    main_tag = alpha_tag(CFG.fed.main_alpha)
    LOG.info("=" * 78)
    LOG.info("HEADLINE (alpha=%s) mean AUC over %d seeds:", main_tag,
             len(CFG.seeds))
    for m, g in rb[rb.alpha == main_tag].groupby("method_name"):
        LOG.info("  %-12s AUC %.4f ± %.4f   Sens@90Spec %.4f ± %.4f",
                 m, g.auc_roc.mean(), g.auc_roc.std(),
                 g.sensitivity_at_90spec.mean(),
                 g.sensitivity_at_90spec.std())
    LOG.info("total wall time %.1f min", (time.time() - t_start) / 60)
    LOG.info("STAGE 03 COMPLETE")


if __name__ == "__main__":
    main()
