"""
Dataset scanning, integrity checking, perceptual-hash de-duplication,
CLAHE enhancement and stratified splitting.

The de-duplication stage is a substantive data-curation contribution: the
released GIED folders contain byte-identical files that appear under *two
different class labels*, which both leaks across train/test and supplies
contradictory ground truth.  We detect and resolve these deterministically.
"""

from __future__ import annotations

import hashlib
import os
import struct
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.fft import dct

from .config import CFG, CLASSES, CLASS_TO_BINARY, CLASS_TO_IDX, RAW_DATA_ROOT

_SOF_MARKERS = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}


# ---------------------------------------------------------------------------
# Integrity
# ---------------------------------------------------------------------------
def jpeg_dimensions(path: Path) -> tuple[int, int] | None:
    """Read (w, h) straight from the JPEG SOF marker without decoding.

    Returns None when the file is not a parseable JPEG - those files are
    quarantined rather than silently fed to the pipeline.
    """
    try:
        with open(path, "rb") as f:
            if f.read(2) != b"\xff\xd8":
                return None
            while True:
                b = f.read(1)
                if not b:
                    return None
                if b != b"\xff":
                    continue
                while b == b"\xff":
                    b = f.read(1)
                marker = b[0]
                if marker in _SOF_MARKERS:
                    f.read(3)
                    h, w = struct.unpack(">HH", f.read(4))
                    return int(w), int(h)
                if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
                    continue
                length = struct.unpack(">H", f.read(2))[0]
                f.seek(length - 2, 1)
    except Exception:
        return None


def md5_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def imread_unicode(path: Path) -> np.ndarray | None:
    """cv2.imread cannot handle the long/unicode Windows paths in this dataset."""
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
        if buf.size == 0:
            return None
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None


def imwrite_unicode(path: Path, img: np.ndarray, quality: int = 95) -> bool:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return False
    buf.tofile(str(path))
    return True


# ---------------------------------------------------------------------------
# Perceptual hashing (DCT pHash, 64-bit)
# ---------------------------------------------------------------------------
def phash(img_bgr: np.ndarray,
          hash_size: int = 8,
          highfreq_factor: int = 4) -> np.uint64:
    """Classic DCT perceptual hash.

    Robust to re-compression, mild rescaling and small brightness shifts -
    exactly the transformations that produce the near-duplicates in GIED.
    """
    n = hash_size * highfreq_factor
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (n, n), interpolation=cv2.INTER_AREA).astype(np.float64)
    coeffs = dct(dct(small, axis=0, norm="ortho"), axis=1, norm="ortho")
    low = coeffs[:hash_size, :hash_size]
    med = np.median(low)
    bits = (low > med).flatten()
    out = np.uint64(0)
    for i, bit in enumerate(bits):
        if bit:
            out |= np.uint64(1) << np.uint64(i)
    return out


def hamming(a: np.uint64, b: np.uint64) -> int:
    return int(bin(int(a) ^ int(b)).count("1"))


def _hash_buckets(codes: np.ndarray, n_bands: int = 4) -> dict:
    """Band-based LSH so we avoid an O(n^2) all-pairs Hamming scan."""
    buckets: dict = defaultdict(list)
    bits_per_band = 64 // n_bands
    for idx, code in enumerate(codes):
        c = int(code)
        for band in range(n_bands):
            key = (band, (c >> (band * bits_per_band)) & ((1 << bits_per_band) - 1))
            buckets[key].append(idx)
    return buckets


def find_near_duplicates(codes: np.ndarray, thresh: int) -> list[tuple[int, int]]:
    """Return candidate index pairs within `thresh` Hamming distance."""
    pairs: set[tuple[int, int]] = set()
    for _, members in _hash_buckets(codes).items():
        if len(members) < 2 or len(members) > 400:
            continue  # skip degenerate mega-buckets (all-black frames etc.)
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                if hamming(codes[a], codes[b]) <= thresh:
                    pairs.add((min(a, b), max(a, b)))
    return sorted(pairs)


class UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


# ---------------------------------------------------------------------------
# Enhancement
# ---------------------------------------------------------------------------
def crop_black_border(img: np.ndarray, thresh: int) -> np.ndarray:
    """Remove the letterbox/vignette border typical of endoscopic capture.

    Operates on the max channel so coloured-but-dark mucosa is not clipped.
    NOTE: this is NOT sufficient on GIED - see `estimate_overlay_free_boxes`.
    """
    gray = img.max(axis=2)
    mask = gray > thresh
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if rows.size < 8 or cols.size < 8:
        return img
    return img[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


# ---------------------------------------------------------------------------
# Burned-in overlay (PHI) localisation and redaction
# ---------------------------------------------------------------------------
# Every GIED frame carries a burned-in processor banner in the black region
# outside the octagonal endoscopic field of view.  On the 720x576 acquisitions
# it is fully legible: "ID No. :", "Name :", "Sex : Age :", "D.O.Birth :", an
# acquisition timestamp ("2022/10/16 11:02:03") and "Comment :".  On the other
# geometries it is cropped to fragments ("Age :", "Birth :", "Patient :", plus
# date and age digits).
#
# This matters twice over.  It is protected health information rendered into
# the pixels, and it is a class-correlated acquisition artefact that a CNN can
# read instead of the mucosa - a textbook shortcut that would silently inflate
# every number in this study.
#
# Detection exploits the fact that the banner is STATIC: it occupies identical
# coordinates in every frame of a given acquisition geometry, so its across-
# image pixel variance is ~0, whereas mucosa varies enormously.  We therefore
# estimate one redaction-safe crop box per native resolution and take the
# largest axis-aligned rectangle that lies wholly inside the variable-content
# region.

def _max_rect_in_histogram(heights: list[int]) -> tuple[int, int, int, int]:
    """Largest rectangle under a histogram -> (area, left, right, height)."""
    stack: list[tuple[int, int]] = []
    best = (0, 0, 0, 0)
    for i in range(len(heights) + 1):
        cur = heights[i] if i < len(heights) else 0
        start = i
        while stack and stack[-1][1] >= cur:
            s, ht = stack.pop()
            area = ht * (i - s)
            if area > best[0]:
                best = (area, s, i, ht)
            start = s
        stack.append((start, cur))
    return best


def largest_inscribed_rect(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Exact maximal axis-aligned rectangle of 1s in a binary mask, O(H*W)."""
    H, W = mask.shape
    heights = np.zeros(W, dtype=np.int32)
    best = (0, 0, 0, 0, 0)
    for y in range(H):
        heights = np.where(mask[y] > 0, heights + 1, 0)
        area, l, r, ht = _max_rect_in_histogram(heights.tolist())
        if area > best[0]:
            best = (area, l, r, y - ht + 1, y + 1)
    _, x0, x1, y0, y1 = best
    return x0, y0, x1 - x0, y1 - y0


def variable_content_mask(gray_stack: np.ndarray,
                          std_thresh: float = 9.0) -> np.ndarray:
    """Pixels whose value actually changes between images of one geometry."""
    std = gray_stack.std(axis=0)
    m = (std > std_thresh).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (11, 11)))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (9, 9)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        m = (lab == largest).astype(np.uint8)
    return m


def estimate_overlay_free_boxes(df: pd.DataFrame,
                                sample_per_group: int = 300,
                                std_thresh: float = 9.0,
                                pad: int = 4,
                                min_group: int = 30,
                                seed: int = 0):
    """One PHI-safe crop box per native resolution.

    Returns (boxes, diagnostics) where `boxes` maps "WxH" -> (x, y, w, h) and
    `diagnostics` carries the variance map and mask for the audit figure.
    Resolution groups with fewer than `min_group` frames cannot support a
    reliable variance estimate and are reported for exclusion instead of being
    cropped with an unvalidated box.
    """
    boxes, diags = {}, {}
    work = df[df.valid] if "valid" in df.columns else df
    for res, grp in work.groupby(work.width_px.astype(str) + "x"
                                 + work.height_px.astype(str)):
        if len(grp) < min_group:
            diags[res] = {"n": len(grp), "box": None, "reason": "too_few_frames"}
            continue
        sam = grp.sample(min(sample_per_group, len(grp)), random_state=seed)
        stack = []
        for p in sam.src_path.tolist():
            im = imread_unicode(Path(p))
            if im is not None:
                stack.append(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32))
        if len(stack) < min_group:
            diags[res] = {"n": len(stack), "box": None, "reason": "unreadable"}
            continue
        arr = np.stack(stack)
        mask = variable_content_mask(arr, std_thresh)
        x, y, w, h = largest_inscribed_rect(mask)
        box = (x + pad, y + pad, max(1, w - 2 * pad), max(1, h - 2 * pad))
        boxes[res] = box
        diags[res] = {"n": len(stack), "box": box, "reason": "ok",
                      "std_map": arr.std(axis=0), "mask": mask,
                      "frame_area": int(arr.shape[1] * arr.shape[2]),
                      "example": sam.iloc[0].src_path}
    return boxes, diags


def apply_box(img: np.ndarray, box) -> np.ndarray:
    x, y, w, h = box
    return img[y:y + h, x:x + w]


def overlay_residual_score(img: np.ndarray) -> float:
    """Fraction of pixels that look like bright glyphs on a dark background.

    Used as an automated post-condition: after redaction this must be ~0.
    Specular mucosal highlights sit on bright neighbourhoods and so do not
    trigger it; overlay text sits on black and does.
    """
    g = img.max(axis=2).astype(np.float32)
    local = cv2.blur(g, (21, 21))
    return float(((g > 140) & (local < 70)).mean())


def apply_clahe(img_bgr: np.ndarray,
                clip: float = 2.0,
                grid: tuple[int, int] = (8, 8)) -> np.ndarray:
    """CLAHE on the L channel of LAB.

    Chroma is deliberately left untouched: mucosal colour and vascular pattern
    are the primary diagnostic cues in white-light endoscopy, and equalising
    a/b would distort them.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
    l2 = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)


def preprocess_image(img_bgr: np.ndarray, box=None, cfg=CFG.preproc) -> np.ndarray:
    """Deterministic chain: PHI-safe crop -> CLAHE -> square resize.

    `box` is the per-resolution redaction box from
    `estimate_overlay_free_boxes`.  When it is None we fall back to the plain
    black-border crop, which is only safe for frames known to carry no banner.
    """
    img = apply_box(img_bgr, box) if box is not None else \
        crop_black_border(img_bgr, cfg.border_crop_thresh)
    img = apply_clahe(img, cfg.clahe_clip_limit, cfg.clahe_tile_grid)
    return cv2.resize(img, (cfg.img_size, cfg.img_size),
                      interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------
@dataclass
class RawRecord:
    image_id: str
    class_label: str
    src_path: str
    filename: str
    width_px: int
    height_px: int
    md5: str
    valid: bool


def scan_raw_dataset(root: Path = RAW_DATA_ROOT) -> pd.DataFrame:
    """Walk the six class folders and build the raw inventory."""
    rows = []
    for cls in CLASSES:
        d = root / cls
        if not d.is_dir():
            raise FileNotFoundError(f"Missing class folder: {d}")
        for fn in sorted(os.listdir(d)):
            p = d / fn
            if not p.is_file():
                continue
            dims = jpeg_dimensions(p)
            valid = dims is not None
            w, h = dims if valid else (-1, -1)
            rows.append(RawRecord(
                image_id=f"{cls.replace(' ', '_')}__{fn}",
                class_label=cls,
                src_path=str(p),
                filename=fn,
                width_px=w,
                height_px=h,
                md5=md5_of(p),
                valid=valid,
            ).__dict__)
    df = pd.DataFrame(rows)
    df["class_idx"] = df["class_label"].map(CLASS_TO_IDX)
    df["binary_label"] = df["class_label"].map(CLASS_TO_BINARY)
    return df


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
def stratified_split(df: pd.DataFrame, cfg=CFG.preproc) -> pd.DataFrame:
    """Stratified train/val/test on the multi-class label.

    Splitting happens *after* de-duplication, so no near-duplicate can span
    the train/test boundary.  The split seed is fixed and independent of the
    method seeds, so every method sees exactly the same test set.
    """
    rng = np.random.RandomState(cfg.split_seed)
    df = df.copy()
    df["split"] = "train"
    for cls in CLASSES:
        idx = df.index[df["class_label"] == cls].to_numpy()
        rng.shuffle(idx)
        n = len(idx)
        n_test = int(round(n * cfg.test_frac))
        n_val = int(round(n * cfg.val_frac))
        df.loc[idx[:n_test], "split"] = "test"
        df.loc[idx[n_test:n_test + n_val], "split"] = "val"
    return df


# ---------------------------------------------------------------------------
# Non-IID client partitioning
# ---------------------------------------------------------------------------
def dirichlet_proportions(n_classes: int, n_clients: int, alpha: float,
                          rng: np.random.RandomState) -> np.ndarray:
    """Draw the class-to-client proportion matrix P[c, k] (rows sum to 1).

    alpha -> 0   : each client sees very few classes (pathological non-IID)
    alpha -> inf : uniform IID split
    """
    if np.isinf(alpha):
        return np.full((n_classes, n_clients), 1.0 / n_clients)
    return np.stack([rng.dirichlet(np.repeat(alpha, n_clients))
                     for _ in range(n_classes)])


def _apply_proportions(labels: np.ndarray, P: np.ndarray,
                       rng: np.random.RandomState) -> list[list[int]]:
    n_classes, n_clients = P.shape
    client_idx: list[list[int]] = [[] for _ in range(n_clients)]
    for c in range(n_classes):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        cuts = (np.cumsum(P[c]) * len(idx)).astype(int)[:-1]
        for cid, part in enumerate(np.split(idx, cuts)):
            client_idx[cid].extend(part.tolist())
    return client_idx


def _repair(client_idx: list[list[int]], labels: np.ndarray,
            min_n: int = 12) -> None:
    """Guarantee every client has enough samples and >=2 classes in situ.

    Without this, a small-alpha draw can leave a client with 0-3 samples, for
    which local training diverges and local AUC is undefined.
    """
    n_classes = int(labels.max()) + 1
    for cid in range(len(client_idx)):
        for _ in range(n_classes):
            own = labels[client_idx[cid]] if client_idx[cid] else np.array([])
            if len(client_idx[cid]) >= min_n and len(set(own.tolist())) >= 2:
                break
            donor = int(np.argmax([len(c) for c in client_idx]))
            if donor == cid or len(client_idx[donor]) < 2 * min_n:
                break
            missing = [c for c in range(n_classes)
                       if c not in set(own.tolist())]
            target_c = missing[0] if missing else int(labels[client_idx[donor][0]])
            pool = [i for i in client_idx[donor] if labels[i] == target_c]
            take = pool[:max(4, len(pool) // 8)] or client_idx[donor][:min_n]
            for t in take:
                client_idx[donor].remove(t)
                client_idx[cid].append(t)


def dirichlet_partition_paired(train_labels: np.ndarray,
                               test_labels: np.ndarray,
                               n_clients: int,
                               alpha: float,
                               seed: int) -> tuple[list, list, np.ndarray]:
    """Partition train AND test with the SAME class-proportion matrix.

    This matters for a fair comparison.  Personalised methods (Local-only,
    FedPer, pFedMe, FedGIM) produce one model per client; each is evaluated on
    that client's own test shard and the predictions are then pooled.  Global
    methods (Centralized, FedAvg, FedProx) predict every test sample with one
    model.  Because the shards tile the test set exactly once, every method
    ends up scored on the identical set of samples, which is what makes the
    paired DeLong and Wilcoxon tests valid.
    """
    rng = np.random.RandomState(seed)
    n_classes = int(max(train_labels.max(), test_labels.max())) + 1
    P = dirichlet_proportions(n_classes, n_clients, alpha, rng)

    tr = _apply_proportions(train_labels, P, rng)
    te = _apply_proportions(test_labels, P, rng)
    _repair(tr, train_labels, min_n=12)
    _repair(te, test_labels, min_n=6)

    tr = [np.array(sorted(c), dtype=np.int64) for c in tr]
    te = [np.array(sorted(c), dtype=np.int64) for c in te]

    # Sanity: shards must tile the test set exactly once.
    allte = np.concatenate(te) if len(te) else np.array([], dtype=np.int64)
    assert len(allte) == len(np.unique(allte)) == len(test_labels), (
        "test shards must partition the test set exactly once")
    return tr, te, P


def dirichlet_partition(labels: np.ndarray, n_clients: int, alpha: float,
                        seed: int) -> list[np.ndarray]:
    """Single-array convenience wrapper around the paired partitioner."""
    rng = np.random.RandomState(seed)
    P = dirichlet_proportions(int(labels.max()) + 1, n_clients, alpha, rng)
    ci = _apply_proportions(labels, P, rng)
    _repair(ci, labels)
    return [np.array(sorted(c), dtype=np.int64) for c in ci]
