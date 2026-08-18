"""Reproducibility, logging and timing helpers."""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from .config import CFG, LOG_DIR


def set_seed(seed: int) -> None:
    """Seed every RNG that can influence a run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.use_deterministic_algorithms(False)  # cuDNN-free CPU run; keep speed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def configure_torch() -> None:
    """Pin thread counts so BLAS cannot oversubscribe this 12-thread laptop."""
    torch.set_num_threads(CFG.n_threads)
    torch.set_num_interop_threads(2)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, str(CFG.n_threads))


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s",
        datefmt="%H:%M:%S",
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(LOG_DIR / f"{name}.log", mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.propagate = False
    return logger


@contextmanager
def timer(label: str, logger: logging.Logger | None = None):
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    msg = f"[{label}] {dt:.2f}s"
    (logger.info(msg) if logger else print(msg))


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def count_params(module: torch.nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in module.parameters()
               if p.requires_grad or not trainable_only)


def state_dict_nbytes(sd: dict) -> int:
    """Bytes on the wire for one model-update transmission."""
    return int(sum(v.numel() * v.element_size() for v in sd.values()
                   if torch.is_tensor(v)))
