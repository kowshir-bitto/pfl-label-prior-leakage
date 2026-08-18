"""
01 - Data preprocessing, integrity audit and de-duplication.

Pipeline
--------
1. Scan the six released class folders; parse JPEG headers without decoding.
2. Quarantine unreadable / non-JPEG files.
3. Exact de-duplication by MD5.
4. Near-duplicate de-duplication by 64-bit DCT perceptual hash (LSH-bucketed).
5. Resolve cross-class conflicts: when a duplicate group spans two class
   labels the ground truth is contradictory, so the ENTIRE group is dropped.
6. Border crop -> CLAHE on LAB-L -> 224x224 resize -> write to data/processed.
7. Stratified 70/10/20 split (fixed seed, shared by every method).

Outputs
-------
    outputs/csv/manifest.csv          one row per retained image
    outputs/csv/dedup_report.csv      every duplicate group and its verdict
    outputs/csv/preprocessing_audit.csv
    outputs/figures/fig02_dataset_overview.{png,pdf}
    outputs/figures/fig03_class_examples_raw.{png,pdf}
    outputs/figures/fig04_class_examples_preprocessed.{png,pdf}
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import (ABNORMAL_CLASSES, CFG, CLASS_SHORT, CLASSES, CSV_DIR,
                        PROC_DIR, RAW_DATA_ROOT)
from src.data_utils import (UnionFind, apply_box, apply_clahe,
                            crop_black_border, estimate_overlay_free_boxes,
                            find_near_duplicates, imread_unicode,
                            imwrite_unicode, overlay_residual_score, phash,
                            preprocess_image, scan_raw_dataset,
                            stratified_split)
from src.plotting import bgr_to_rgb, panel_label, save_fig, use_paper_style
from src.utils import get_logger, set_seed, timer

LOG = get_logger("01_preprocess")
PP = CFG.preproc


# ---------------------------------------------------------------------------
def build_inventory() -> pd.DataFrame:
    LOG.info("Scanning raw dataset at %s", RAW_DATA_ROOT)
    with timer("scan", LOG):
        df = scan_raw_dataset()
    LOG.info("Found %d files across %d classes", len(df), df.class_label.nunique())
    n_bad = int((~df.valid).sum())
    if n_bad:
        LOG.warning("QUARANTINE: %d files are not parseable JPEG", n_bad)
        for _, r in df[~df.valid].iterrows():
            LOG.warning("   corrupt -> %s/%s", r.class_label, r.filename)
    return df


def compute_phashes(df: pd.DataFrame) -> pd.DataFrame:
    """Decode once, store the pHash and the decoded-size sanity check."""
    codes, decoded_ok = [], []
    for p in tqdm(df.src_path.tolist(), desc="pHash", ncols=90):
        img = imread_unicode(Path(p))
        if img is None:
            codes.append(np.uint64(0))
            decoded_ok.append(False)
            continue
        codes.append(phash(img, PP.phash_size, PP.phash_highfreq_factor))
        decoded_ok.append(True)
    df = df.copy()
    df["phash"] = codes
    df["decodable"] = decoded_ok
    return df


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Group exact + near duplicates, keep one survivor per group.

    Verdicts
    --------
    kept                 the survivor of a same-class duplicate group
    dropped_duplicate    a redundant member of a same-class group
    dropped_conflict     member of a group whose labels disagree -> unusable
    """
    work = df[df.valid & df.decodable].reset_index(drop=True)
    n = len(work)
    uf = UnionFind(n)

    # ---- exact duplicates (MD5) -----------------------------------------
    by_md5 = defaultdict(list)
    for i, m in enumerate(work.md5.tolist()):
        by_md5[m].append(i)
    n_exact_groups = 0
    for members in by_md5.values():
        if len(members) > 1:
            n_exact_groups += 1
            for j in members[1:]:
                uf.union(members[0], j)
    LOG.info("Exact (MD5) duplicate groups: %d", n_exact_groups)

    # ---- near duplicates (pHash + LSH) ----------------------------------
    codes = work.phash.to_numpy()
    with timer("near-duplicate search", LOG):
        pairs = find_near_duplicates(codes, PP.phash_hamming_thresh)
    LOG.info("Near-duplicate candidate pairs (Hamming<=%d): %d",
             PP.phash_hamming_thresh, len(pairs))
    for a, b in pairs:
        uf.union(a, b)

    # ---- resolve groups --------------------------------------------------
    groups = defaultdict(list)
    for i in range(n):
        groups[uf.find(i)].append(i)

    verdict = np.array(["kept"] * n, dtype=object)
    group_id = np.full(n, -1, dtype=np.int64)
    rows = []
    gid = 0
    for _, members in groups.items():
        if len(members) == 1:
            continue
        labels = {work.class_label.iloc[i] for i in members}
        conflict = len(labels) > 1
        for i in members:
            group_id[i] = gid
        if conflict:
            for i in members:
                verdict[i] = "dropped_conflict"
        else:
            # deterministic survivor: highest pixel count, then name order
            order = sorted(members,
                           key=lambda i: (-(work.width_px.iloc[i] *
                                            work.height_px.iloc[i]),
                                          work.filename.iloc[i]))
            for i in order[1:]:
                verdict[i] = "dropped_duplicate"
        rows.append({
            "group_id": gid,
            "group_size": len(members),
            "labels_in_group": " | ".join(sorted(labels)),
            "cross_class_conflict": conflict,
            "members": " ; ".join(
                f"{work.class_label.iloc[i]}/{work.filename.iloc[i]}"
                for i in members),
            "verdict": "ALL DROPPED (contradictory labels)" if conflict
                       else "kept 1, dropped %d" % (len(members) - 1),
        })
        gid += 1

    work["dup_group"] = group_id
    work["dedup_verdict"] = verdict
    report = pd.DataFrame(rows)

    n_conf = int(report.cross_class_conflict.sum()) if len(report) else 0
    LOG.warning("Duplicate groups total: %d  |  CROSS-CLASS CONFLICTS: %d",
                len(report), n_conf)
    LOG.info("Dropped as duplicate: %d | dropped as conflict: %d",
             int((verdict == "dropped_duplicate").sum()),
             int((verdict == "dropped_conflict").sum()))
    return work, report


def write_processed(df: pd.DataFrame, boxes: dict) -> pd.DataFrame:
    """PHI-safe crop -> CLAHE -> resize -> save, auditing overlay removal."""
    for c in CLASSES:
        (PROC_DIR / c.replace(" ", "_")).mkdir(parents=True, exist_ok=True)

    out_paths, crop_w, crop_h, ok_flags = [], [], [], []
    res_before, res_after = [], []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="preprocess", ncols=90):
        img = imread_unicode(Path(r.src_path))
        if img is None:
            out_paths.append(""); crop_w.append(-1); crop_h.append(-1)
            ok_flags.append(False); res_before.append(np.nan)
            res_after.append(np.nan)
            continue
        box = boxes.get(f"{r.width_px}x{r.height_px}")
        cropped = apply_box(img, box) if box else crop_black_border(
            img, PP.border_crop_thresh)
        crop_h.append(cropped.shape[0]); crop_w.append(cropped.shape[1])
        res_before.append(overlay_residual_score(img))
        res_after.append(overlay_residual_score(cropped))
        proc = cv2.resize(
            apply_clahe(cropped, PP.clahe_clip_limit, PP.clahe_tile_grid),
            (PP.img_size, PP.img_size), interpolation=cv2.INTER_AREA)
        dst = PROC_DIR / r.class_label.replace(" ", "_") / f"{r.proc_name}.jpg"
        ok_flags.append(imwrite_unicode(dst, proc))
        out_paths.append(str(dst))

    df = df.copy()
    df["proc_path"] = out_paths
    df["cropped_w"] = crop_w
    df["cropped_h"] = crop_h
    df["written"] = ok_flags
    df["overlay_residual_raw"] = res_before
    df["overlay_residual_proc"] = res_after
    return df


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig02_dataset_overview(raw: pd.DataFrame, final: pd.DataFrame) -> None:
    """Fig 2 - dataset composition, curation effect, acquisition geometry."""
    use_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.2))

    # (a) per-class counts, raw vs retained, with the binary task shaded
    ax = axes[0]
    raw_counts = [int((raw.class_label == c).sum()) for c in CLASSES]
    fin_counts = [int((final.class_label == c).sum()) for c in CLASSES]
    x = np.arange(len(CLASSES)); w = 0.38
    b1 = ax.bar(x - w / 2, raw_counts, w, label="Released",
                color="#B0BEC5", edgecolor="black", linewidth=0.4)
    b2 = ax.bar(x + w / 2, fin_counts, w, label="After curation",
                color=[CFG.fig.proposed_color if c in ABNORMAL_CLASSES
                       else CFG.fig.baseline_color for c in CLASSES],
                edgecolor="black", linewidth=0.4)
    for rect, v in zip(b1, raw_counts):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 40, str(v),
                ha="center", fontsize=5.5, color="#455A64")
    for rect, v in zip(b2, fin_counts):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 40, str(v),
                ha="center", fontsize=5.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([CLASS_SHORT[c] for c in CLASSES], rotation=30,
                       ha="right")
    ax.set_ylabel("Number of images")
    ax.set_title("Class distribution", fontweight="bold")
    ax.set_ylim(0, max(raw_counts) * 1.18)
    ax.legend(loc="upper left", fontsize=6)
    imb = max(fin_counts) / max(min(fin_counts), 1)
    ax.text(0.98, 0.72, f"imbalance\n{imb:.1f}Ã—", transform=ax.transAxes,
            ha="right", fontsize=6.5, style="italic", color="#37474F")
    panel_label(ax, "a")

    # (b) curation waterfall
    ax = axes[1]
    n_raw = len(raw)
    n_corrupt = int((~raw.valid).sum())
    n_dup = int((final.attrs.get("n_dropped_dup", 0)))
    n_conf = int((final.attrs.get("n_dropped_conflict", 0)))
    n_final = len(final)
    stages = ["Released", "Corrupt", "Near-dup", "Label\nconflict", "Retained"]
    vals = [n_raw, -n_corrupt, -n_dup, -n_conf, n_final]
    colors = ["#B0BEC5", "#EF6C00", "#EF6C00", "#C62828", "#2E7D32"]
    running = 0
    for i, (s, v, c) in enumerate(zip(stages, vals, colors)):
        if i in (0, len(stages) - 1):
            ax.bar(i, abs(v), color=c, edgecolor="black", linewidth=0.4)
            ax.text(i, abs(v) + 90, f"{abs(v)}", ha="center",
                    fontsize=6.5, fontweight="bold")
            running = abs(v) if i == 0 else running
        else:
            ax.bar(i, abs(v), bottom=running + v, color=c,
                   edgecolor="black", linewidth=0.4)
            ax.text(i, running + 120, f"âˆ’{abs(v)}", ha="center",
                    fontsize=6.5, fontweight="bold", color="#C62828")
            running += v
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stages, fontsize=6.5)
    ax.set_ylabel("Number of images")
    ax.set_title("Curation waterfall", fontweight="bold")
    panel_label(ax, "b")

    # (c) pre-resize acquisition geometry
    ax = axes[2]
    dims = (raw[raw.valid]
            .assign(res=lambda d: d.width_px.astype(str) + "Ã—" +
                    d.height_px.astype(str))
            .res.value_counts())
    ax.barh(range(len(dims)), dims.values, color=CFG.fig.baseline_color,
            edgecolor="black", linewidth=0.4)
    ax.set_yticks(range(len(dims)))
    ax.set_yticklabels(dims.index, fontsize=6.5)
    ax.invert_yaxis()
    for i, v in enumerate(dims.values):
        ax.text(v + 60, i, f"{v}  ({100*v/dims.sum():.1f}%)",
                va="center", fontsize=6)
    ax.set_xlabel("Number of images")
    ax.set_title("Native acquisition resolution", fontweight="bold")
    ax.set_xlim(0, dims.max() * 1.32)
    panel_label(ax, "c")

    save_fig(fig, "fig02_dataset_overview")
    LOG.info("wrote fig02_dataset_overview")


def _sample_paths(df: pd.DataFrame, n: int, seed: int = 7) -> dict:
    rng = np.random.RandomState(seed)
    out = {}
    for c in CLASSES:
        sub = df[df.class_label == c]
        idx = rng.choice(len(sub), size=min(n, len(sub)), replace=False)
        out[c] = sub.iloc[np.sort(idx)]
    return out


def _redact_for_display(img: np.ndarray, box) -> np.ndarray:
    """Darken everything outside the PHI-safe box so raw frames can be shown.

    Publishing an un-redacted GIED frame would reproduce burned-in patient
    identifiers, so no figure in this project ever renders the raw banner.
    """
    if box is None:
        return img
    x, y, w, h = box
    out = (img * 0.16).astype(np.uint8)
    out[y:y + h, x:x + w] = img[y:y + h, x:x + w]
    cv2.rectangle(out, (x, y), (x + w, y + h), (0, 230, 255), 2)
    return out


def fig03_examples_raw(df: pd.DataFrame, boxes: dict,
                       n_per_class: int = 5) -> None:
    """Fig 3 - representative frames per class, banner region redacted."""
    use_paper_style()
    samples = _sample_paths(df, n_per_class)
    fig, axes = plt.subplots(len(CLASSES), n_per_class,
                             figsize=(1.55 * n_per_class, 1.62 * len(CLASSES)))
    for r, c in enumerate(CLASSES):
        sub = samples[c]
        for k in range(n_per_class):
            ax = axes[r, k]
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            if k < len(sub):
                rec = sub.iloc[k]
                img = imread_unicode(Path(rec.src_path))
                if img is not None:
                    box = boxes.get(f"{rec.width_px}x{rec.height_px}")
                    ax.imshow(bgr_to_rgb(_redact_for_display(img, box)))
                    if k == 0:
                        ax.set_ylabel(CLASS_SHORT[c], fontsize=8,
                                      fontweight="bold", rotation=0,
                                      ha="right", va="center", labelpad=26)
                    ax.text(0.5, -0.07,
                            f"{img.shape[1]}Ã—{img.shape[0]}",
                            transform=ax.transAxes, ha="center", va="top",
                            fontsize=5.2, color="#546E7A")
            for s in ax.spines.values():
                s.set_visible(True); s.set_linewidth(0.6)
                s.set_edgecolor(CFG.fig.proposed_color
                                if c in ABNORMAL_CLASSES
                                else CFG.fig.baseline_color)
    fig.suptitle("Representative endoscopic frames per class  "
                 "(orange = abnormal, blue = normal; dimmed area = burned-in "
                 "banner region, redacted)",
                 fontsize=8.5, fontweight="bold", y=0.995)
    save_fig(fig, "fig03_class_examples_raw")
    LOG.info("wrote fig03_class_examples_raw")


def fig04_examples_preprocessed(df: pd.DataFrame, boxes: dict) -> None:
    """Fig 4 - the preprocessing chain, per class."""
    use_paper_style()
    samples = _sample_paths(df, 1, seed=11)
    stages = ["Acquired frame\n(banner redacted)", "PHI-safe FOV crop",
              "CLAHE (LAB-L)", "Resized 224Ã—224"]
    fig, axes = plt.subplots(len(CLASSES), len(stages),
                             figsize=(1.68 * len(stages), 1.68 * len(CLASSES)))
    for r, c in enumerate(CLASSES):
        rec = samples[c].iloc[0]
        raw = imread_unicode(Path(rec.src_path))
        if raw is None:
            continue
        box = boxes.get(f"{rec.width_px}x{rec.height_px}")
        shown = _redact_for_display(raw, box)
        cropped = apply_box(raw, box) if box else crop_black_border(
            raw, PP.border_crop_thresh)
        clahed = apply_clahe(cropped, PP.clahe_clip_limit, PP.clahe_tile_grid)
        final = cv2.resize(clahed, (PP.img_size, PP.img_size),
                           interpolation=cv2.INTER_AREA)
        for k, im in enumerate([shown, cropped, clahed, final]):
            ax = axes[r, k]
            ax.imshow(bgr_to_rgb(im))
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            if r == 0:
                ax.set_title(stages[k], fontsize=7, fontweight="bold")
            if k == 0:
                ax.set_ylabel(CLASS_SHORT[c], fontsize=8, fontweight="bold",
                              rotation=0, ha="right", va="center", labelpad=26)
            ax.text(0.5, -0.06, f"{im.shape[1]}Ã—{im.shape[0]}",
                    transform=ax.transAxes, ha="center", va="top",
                    fontsize=5.2, color="#546E7A")
            for s in ax.spines.values():
                s.set_visible(True); s.set_linewidth(0.6)
                s.set_edgecolor("#CFD8DC")
    fig.suptitle("Deterministic preprocessing chain applied to every image",
                 fontsize=9, fontweight="bold", y=0.997)
    save_fig(fig, "fig04_class_examples_preprocessed")
    LOG.info("wrote fig04_class_examples_preprocessed")


def fig05_phi_audit(diags: dict, df: pd.DataFrame) -> None:
    """Fig 5 - how the burned-in banner is localised and removed.

    Left column: across-image pixel standard deviation per acquisition
    geometry.  The banner is static, so it appears as a zero-variance stencil
    of the text; the octagonal field of view appears as the high-variance
    region.  Right: the resulting crop and the residual-overlay audit.
    """
    use_paper_style()
    keys = [k for k, v in diags.items() if v.get("box")]
    n = len(keys)
    fig, axes = plt.subplots(n, 3, figsize=(9.4, 2.9 * n))
    axes = np.atleast_2d(axes)
    for i, res in enumerate(keys):
        d = diags[res]
        x, y, w, h = d["box"]
        ax = axes[i, 0]
        ax.imshow(d["std_map"], cmap="inferno")
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, lw=1.4,
                                   edgecolor="#00E5FF"))
        ax.set_title(f"{res}  (n={d['n']})\nacross-image SD", fontsize=7.5)
        ax.set_xticks([]); ax.set_yticks([])
        if i == 0:
            panel_label(ax, "a", dx=-0.03)

        ax = axes[i, 1]
        ax.imshow(d["mask"], cmap="gray")
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, lw=1.4,
                                   edgecolor="#00E5FF"))
        ax.set_title("variable-content mask\n+ maximal inscribed rectangle",
                     fontsize=7.5)
        ax.set_xticks([]); ax.set_yticks([])
        if i == 0:
            panel_label(ax, "b", dx=-0.03)

        ax = axes[i, 2]
        ex = imread_unicode(Path(d["example"]))
        if ex is not None:
            ax.imshow(bgr_to_rgb(apply_box(ex, d["box"])))
        keep = 100.0 * w * h / d["frame_area"]
        ax.set_title(f"redacted crop {w}Ã—{h}\n({keep:.0f}% of frame retained)",
                     fontsize=7.5)
        ax.set_xticks([]); ax.set_yticks([])
        if i == 0:
            panel_label(ax, "c", dx=-0.03)

    fig.suptitle("Localisation and removal of burned-in acquisition banners "
                 "(patient identifiers)", fontsize=9.5, fontweight="bold",
                 y=0.998)
    save_fig(fig, "fig05_phi_redaction_audit")
    LOG.info("wrote fig05_phi_redaction_audit")


# ---------------------------------------------------------------------------
def main() -> None:
    set_seed(PP.split_seed)

    raw = build_inventory()
    raw.to_csv(CSV_DIR / "raw_inventory.csv", index=False)

    raw = compute_phashes(raw)
    work, report = deduplicate(raw)
    report.to_csv(CSV_DIR / "dedup_report.csv", index=False)

    n_dup = int((work.dedup_verdict == "dropped_duplicate").sum())
    n_conf = int((work.dedup_verdict == "dropped_conflict").sum())
    keep = work[work.dedup_verdict == "kept"].reset_index(drop=True)

    # ---- burned-in banner (PHI) localisation --------------------------
    LOG.info("estimating PHI-safe crop boxes from across-image variance ...")
    boxes, diags = estimate_overlay_free_boxes(keep)
    for res, d in diags.items():
        if d.get("box"):
            x, y, w, h = d["box"]
            LOG.info("  %-9s n=%-5d box=(%d,%d,%d,%d) retains %.1f%%",
                     res, d["n"], x, y, w, h,
                     100.0 * w * h / d["frame_area"])
        else:
            LOG.warning("  %-9s n=%-5d NO BOX (%s) -> frames excluded",
                        res, d["n"], d["reason"])
    pd.DataFrame([
        {"resolution": r, "n_sampled": d["n"], "reason": d["reason"],
         "box_x": (d["box"] or [None] * 4)[0], "box_y": (d["box"] or [None] * 4)[1],
         "box_w": (d["box"] or [None] * 4)[2], "box_h": (d["box"] or [None] * 4)[3]}
        for r, d in diags.items()]).to_csv(
        CSV_DIR / "phi_redaction_boxes.csv", index=False)

    # Frames whose geometry has too few examples to validate a box are
    # excluded rather than shipped with unverified redaction.
    res_key = keep.width_px.astype(str) + "x" + keep.height_px.astype(str)
    unsupported = ~res_key.isin(boxes.keys())
    n_unsup = int(unsupported.sum())
    if n_unsup:
        LOG.warning("excluding %d frames whose geometry has no validated "
                    "redaction box", n_unsup)
        keep = keep[~unsupported].reset_index(drop=True)

    keep["proc_name"] = [
        f"{r.class_label.replace(' ', '_')}_{i:05d}"
        for i, r in enumerate(keep.itertuples())
    ]

    keep = write_processed(keep, boxes)
    keep = keep[keep.written].reset_index(drop=True)
    keep = stratified_split(keep)
    keep.attrs["n_dropped_dup"] = n_dup
    keep.attrs["n_dropped_conflict"] = n_conf

    cols = ["image_id", "proc_name", "class_label", "class_idx",
            "binary_label", "split", "src_path", "proc_path", "filename",
            "width_px", "height_px", "cropped_w", "cropped_h", "md5",
            "dup_group", "overlay_residual_raw", "overlay_residual_proc"]
    keep[cols].to_csv(CSV_DIR / "manifest.csv", index=False)

    n_raw_flag = int((keep.overlay_residual_raw > 1e-3).sum())
    n_proc_flag = int((keep.overlay_residual_proc > 1e-3).sum())
    audit = pd.DataFrame([{
        "released_total": len(raw),
        "corrupt_quarantined": int((~raw.valid).sum()),
        "dropped_near_duplicate": n_dup,
        "dropped_label_conflict": n_conf,
        "dropped_no_redaction_box": n_unsup,
        "frames_with_overlay_before": n_raw_flag,
        "frames_with_overlay_after": n_proc_flag,
        "mean_overlay_residual_before": float(keep.overlay_residual_raw.mean()),
        "mean_overlay_residual_after": float(keep.overlay_residual_proc.mean()),
        "retained_total": len(keep),
        "n_train": int((keep.split == "train").sum()),
        "n_val": int((keep.split == "val").sum()),
        "n_test": int((keep.split == "test").sum()),
        "n_abnormal": int((keep.binary_label == 1).sum()),
        "n_normal": int((keep.binary_label == 0).sum()),
        "imbalance_ratio_multiclass": round(
            keep.class_label.value_counts().max() /
            keep.class_label.value_counts().min(), 3),
    }])
    audit.to_csv(CSV_DIR / "preprocessing_audit.csv", index=False)

    LOG.info("=" * 72)
    for k, v in audit.iloc[0].items():
        LOG.info("%-26s %s", k, v)
    LOG.info("split counts per class:\n%s",
             pd.crosstab(keep.class_label, keep.split))
    LOG.info("=" * 72)

    LOG.info("PHI AUDIT: frames with detectable burned-in overlay "
             "%d -> %d after redaction (mean residual %.5f -> %.5f)",
             n_raw_flag, n_proc_flag,
             keep.overlay_residual_raw.mean(), keep.overlay_residual_proc.mean())

    fig02_dataset_overview(raw, keep)
    fig03_examples_raw(keep, boxes)
    fig04_examples_preprocessed(keep, boxes)
    fig05_phi_audit(diags, keep)
    LOG.info("STAGE 01 COMPLETE")


if __name__ == "__main__":
    main()
