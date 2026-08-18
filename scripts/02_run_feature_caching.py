"""
02 - Frozen-encoder feature bank.

Runs the two frozen ImageNet encoders (ResNet-18 + EfficientNet-B0) exactly
once over the preprocessed corpus and caches the concatenated 1792-d
penultimate embeddings.  Every downstream federated experiment then trains on
these cached vectors, which is what makes 7 methods x 5 seeds x 4 Dirichlet
settings tractable on a CPU-only machine.

Views
-----
train : 3 stochastic augmented views  -> augmentation diversity for the head
val   : 3 deterministic views         -> reference window for drift detection
test  : 3 deterministic views (identity / hflip / vflip)
                                      -> genuine multi-view test-time adaptation

Outputs
-------
    data/cache/feat_{split}.npy      float32 (N, V, D)
    data/cache/meta_{split}.csv      row-aligned labels and identifiers
    data/cache/feature_bank_info.json
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.config import CACHE_DIR, CFG, CSV_DIR
from src.data_utils import imread_unicode
from src.models import MultiEncoder
from src.utils import configure_torch, get_logger, set_seed, timer
from src.xai import to_tensor

LOG = get_logger("02_features")
MC = CFG.model


# ---------------------------------------------------------------------------
class ViewDataset(Dataset):
    """Yields V views of each preprocessed 224x224 image as a (V,3,H,W) tensor."""

    def __init__(self, paths: list[str], n_views: int, train: bool, seed: int):
        self.paths = paths
        self.n_views = n_views
        self.train = train
        self.seed = seed

    def __len__(self) -> int:
        return len(self.paths)

    def _augment(self, img: np.ndarray, v: int, idx: int) -> np.ndarray:
        """View 0 is always the untouched image so it is directly comparable
        across splits; further views are augmentations."""
        if v == 0:
            return img
        if not self.train:
            # Deterministic TTA views: flips only. Reproducible, label-safe,
            # and anatomically plausible for endoscopic frames.
            return img[:, ::-1].copy() if v == 1 else img[::-1, :].copy()

        rng = np.random.RandomState((self.seed * 1_000_003 + idx * 17 + v)
                                    % (2 ** 31 - 1))
        out = img
        if rng.rand() < 0.5:
            out = out[:, ::-1].copy()
        if rng.rand() < 0.3:
            out = out[::-1, :].copy()
        # random resized crop (scale 0.80-1.00)
        s = rng.uniform(0.80, 1.0)
        h, w = out.shape[:2]
        ch, cw = int(h * s), int(w * s)
        y0 = rng.randint(0, h - ch + 1)
        x0 = rng.randint(0, w - cw + 1)
        out = out[y0:y0 + ch, x0:x0 + cw]
        import cv2
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
        # mild photometric jitter (endoscopic illumination varies shot to shot)
        gain = rng.uniform(0.90, 1.10)
        bias = rng.uniform(-10, 10)
        return np.clip(out.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)

    def __getitem__(self, i: int):
        img = imread_unicode(Path(self.paths[i]))
        if img is None:
            img = np.zeros((CFG.preproc.img_size, CFG.preproc.img_size, 3),
                           dtype=np.uint8)
        views = [to_tensor(self._augment(img, v, i)) for v in range(self.n_views)]
        return torch.stack(views, 0), i


# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_split(encoder: MultiEncoder, df: pd.DataFrame, split: str,
                  n_views: int, train: bool) -> np.ndarray:
    ds = ViewDataset(df.proc_path.tolist(), n_views, train,
                     CFG.preproc.split_seed)
    dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=4,
                    pin_memory=False, persistent_workers=False)
    D = encoder.out_dim
    out = np.zeros((len(ds), n_views, D), dtype=np.float32)
    for batch, idxs in tqdm(dl, desc=f"encode/{split}", ncols=92):
        B, V = batch.shape[0], batch.shape[1]
        feats = encoder(batch.view(B * V, *batch.shape[2:]))
        out[idxs.numpy()] = feats.view(B, V, D).numpy()
    return out


def main() -> None:
    configure_torch()
    set_seed(CFG.preproc.split_seed)

    man = pd.read_csv(CSV_DIR / "manifest.csv")
    LOG.info("manifest: %d images | splits: %s", len(man),
             man.split.value_counts().to_dict())

    LOG.info("building frozen encoders: %s", list(MC.backbones))
    encoder = MultiEncoder(MC.backbones, pretrained=True)
    encoder.eval()
    LOG.info("concatenated embedding dim = %d", encoder.out_dim)

    info = {"backbones": list(MC.backbones), "dim": int(encoder.out_dim),
            "views": {}, "counts": {}}

    for split, n_views, is_train in (("train", MC.train_views, True),
                                     ("val", MC.test_views, False),
                                     ("test", MC.test_views, False)):
        sub = man[man.split == split].reset_index(drop=True)
        LOG.info("--- %s: %d images x %d views ---", split, len(sub), n_views)
        with timer(f"encode {split}", LOG):
            feats = extract_split(encoder, sub, split, n_views, is_train)
        np.save(CACHE_DIR / f"feat_{split}.npy", feats)
        sub[["image_id", "proc_name", "class_label", "class_idx",
             "binary_label", "proc_path"]].to_csv(
            CACHE_DIR / f"meta_{split}.csv", index=False)
        info["views"][split] = n_views
        info["counts"][split] = int(len(sub))
        LOG.info("saved feat_%s.npy shape=%s (%.1f MB)", split, feats.shape,
                 feats.nbytes / 1024 ** 2)

    import json
    with open(CACHE_DIR / "feature_bank_info.json", "w") as f:
        json.dump(info, f, indent=2)
    LOG.info("STAGE 02 COMPLETE  |  %s", info)


if __name__ == "__main__":
    main()
