"""
Global configuration for the FedGIM study.

FedGIM = Federated Gastro-endoscopic Inference Management
    "An Inference Management Framework for Federated Collaborative
     Gastrointestinal Endoscopic Screening under Non-IID Distribution"

Every experimental constant lives here so that any run is reproducible from
this single file.  Scripts import `from src.config import CFG`.

IMPORTANT DESIGN NOTES (also reported verbatim in the paper's Methods):
  * The GIED release is single-source.  Multi-institutional federation is
    *simulated* via Dirichlet partitioning; this is stated explicitly and is
    standard practice in the FL literature.
  * The release contains endoscopy only.  No CT data exists, so the study is
    single-modality.  The `modality` column is retained in all CSV schemas for
    forward compatibility but is always "endoscopy".
  * Training runs on a CPU-only edge-class machine.  All methods therefore use
    parameter-efficient federated learning: a frozen, publicly pre-trained
    encoder plus a federated adapter head.  This is applied identically to
    every baseline and to the proposed method, so comparisons are fair.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
# Both roots are configurable so the pipeline runs unchanged on any machine:
#
#   GIED_PROJECT_ROOT
#                   where derived data, checkpoints and outputs are written.
#                   Defaults to the repository itself.
#   GIED_RAW_ROOT   directory holding the five class subdirectories of the GIED
#                   release (Cancer, Gerd, Gerd Normal, Polyp, Polyp Normal).
#                   Defaults to `<project>/data/raw`.  Read, never written to.
#
# Keep the project root short on Windows: raw image paths in the release run to
# 259 characters, one under MAX_PATH, so a long output root can push derived
# paths over the limit.
PROJECT_ROOT = Path(
    os.environ.get("GIED_PROJECT_ROOT", Path(__file__).resolve().parents[1])
).expanduser()
DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_ROOT = Path(
    os.environ.get("GIED_RAW_ROOT", DATA_DIR / "raw")).expanduser()
PROC_DIR = DATA_DIR / "processed"          # CLAHE'd, resized 224x224 JPEGs
CACHE_DIR = DATA_DIR / "cache"             # memmapped frozen-encoder features
OUT_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
TAB_DIR = OUT_DIR / "tables"
CSV_DIR = OUT_DIR / "csv"
LOG_DIR = OUT_DIR / "logs"
CKPT_DIR = OUT_DIR / "checkpoints"

for _d in (DATA_DIR, PROC_DIR, CACHE_DIR, OUT_DIR, FIG_DIR, TAB_DIR, CSV_DIR,
           LOG_DIR, CKPT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Label taxonomy
# --------------------------------------------------------------------------
# Five classes as present in the corrected GIED release.  An earlier release
# carried a sixth class ("Spot") that was withdrawn as mislabelled; it is
# absent here and no result in this study involves it.
CLASSES = ["Cancer", "Gerd", "Gerd Normal", "Polyp", "Polyp Normal"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
N_CLASSES = len(CLASSES)

# Short display names for figures (long names wreck tick labels).
CLASS_SHORT = {
    "Cancer": "Cancer",
    "Gerd": "GERD",
    "Gerd Normal": "GERD-Nrm",
    "Polyp": "Polyp",
    "Polyp Normal": "Polyp-Nrm",
}

# PRIMARY TASK - binary screening (drives AUC, Sens@90%Spec, DeLong, ROC, HITL).
# "Abnormal" = any pathological finding requiring clinician review.
ABNORMAL_CLASSES = ["Cancer", "Gerd", "Polyp"]                  # 2267 images
NORMAL_CLASSES = ["Gerd Normal", "Polyp Normal"]                # 2331 images
BINARY_POSITIVE = 1  # 1 == Abnormal == the class we must not miss
CLASS_TO_BINARY = {c: (1 if c in ABNORMAL_CLASSES else 0) for c in CLASSES}
BINARY_NAMES = ["Normal", "Abnormal"]

# SECONDARY TASK - 5-class (drives confusion matrices, macro-F1, subgroups).
# `lesion_stage` strata for subgroup_lesion_analysis.csv are the abnormal
# classes; normals are pooled as a single "Normal" stratum.
LESION_STRATA = ["Cancer", "Gerd", "Polyp", "Normal"]

MODALITY = "endoscopy"   # single modality; no CT exists in this release


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
SEEDS = [42, 123, 456, 789, 2024]


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class PreprocCfg:
    img_size: int = 224
    # Perceptual-hash dedup.  8x8 DCT pHash -> 64-bit code; Hamming distance
    # <= threshold means "near duplicate".
    phash_size: int = 8
    phash_highfreq_factor: int = 4
    phash_hamming_thresh: int = 5
    # CLAHE is applied to the L channel of LAB so chroma (mucosal colour, the
    # main diagnostic cue in endoscopy) is left untouched.
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: tuple = (8, 8)
    # Endoscopic frames carry a black border/letterbox.  Crop it before resize.
    border_crop_thresh: int = 12
    # Split fractions (stratified on the multi-class label).
    train_frac: float = 0.70
    val_frac: float = 0.10
    test_frac: float = 0.20
    split_seed: int = 42          # split is FIXED across all method seeds


# --------------------------------------------------------------------------
# Frozen encoders + federated adapter head
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelCfg:
    # Two ImageNet-pretrained encoders, frozen.  Their concatenated penultimate
    # features form the representation every FL method operates on.
    backbones: tuple = ("resnet18", "efficientnet_b0")
    feat_dims: dict = field(default_factory=lambda: {
        "resnet18": 512, "efficientnet_b0": 1280})
    # Number of augmented views cached per training image (feature-space
    # augmentation diversity) and per test image (for multi-view TTA).
    train_views: int = 3
    test_views: int = 3
    # Federated adapter head: Linear -> BN -> GELU -> Dropout -> Linear.
    # BN is what TENT-style test-time adaptation updates.
    hidden_dim: int = 256
    dropout: float = 0.3

    @property
    def total_feat_dim(self) -> int:
        return sum(self.feat_dims[b] for b in self.backbones)

    # ImageNet normalisation for both encoders.
    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------
# Federated learning
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class FedCfg:
    n_clients: int = 5
    rounds: int = 20
    local_epochs: int = 2
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    # Dirichlet concentration.  `inf` == perfectly IID partition.
    alphas: tuple = (0.1, 0.5, 1.0, float("inf"))
    main_alpha: float = 0.5       # headline non-IID setting
    # FedProx proximal coefficient.
    mu_prox: float = 0.01
    # pFedMe.  The outer step is eta*lambda; tying eta to the head lr (1e-3)
    # made it 0.015, so w barely moved toward theta and the method looked
    # broken.  Kept explicit so the effective step is visible.
    pfedme_lambda: float = 15.0
    pfedme_k_steps: int = 5
    pfedme_beta: float = 1.0
    pfedme_outer_lr: float = 1e-2      # eta*lambda = 0.15
    # FedPer: how many trailing modules stay client-private (never averaged).
    fedper_private_suffix: tuple = ("fc2",)
    # Class-balanced loss (Cui et al. effective number) used by FedGIM.
    cb_beta: float = 0.999


METHODS = [
    "Local-only",     # each client trains alone, no communication
    "Centralized",    # pooled data upper bound (privacy-violating reference)
    "FedAvg",
    "FedProx",
    "FedPer",
    "pFedMe",
    "FedGIM",         # PROPOSED
]
PROPOSED_METHOD = "FedGIM"
# The IM layer is model-agnostic; we also bolt it onto FedAvg to isolate its
# contribution independently of the FedGIM training recipe.
IM_TRANSFER_BASE = "FedAvg"


# --------------------------------------------------------------------------
# Inference Management layer - the paper's central contribution
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class IMCfg:
    # (1) Privacy-preserving inference: Gaussian mechanism on released logits.
    dp_enabled: bool = True
    dp_epsilon: float = 4.0
    dp_delta: float = 1e-5
    dp_sensitivity: float = 1.0
    dp_epsilon_sweep: tuple = (0.5, 1.0, 2.0, 4.0, 8.0, float("inf"))

    # (2) Test-time adaptation: TENT-style entropy minimisation on head BN
    #     affine params + multi-view TTA averaging.
    tta_enabled: bool = True
    tta_lr: float = 1e-3
    tta_steps: int = 3
    tta_views: int = 3

    # (3) Resource-aware deployment: INT8 dynamic quantisation of the head.
    compress_enabled: bool = True
    quant_dtype: str = "qint8"

    # (4) Drift detection: MMD (RBF) + per-dimension KS on encoder feature
    #     statistics against a stored reference window.
    drift_enabled: bool = True
    drift_window: int = 256
    drift_mmd_thresh: float = 0.05
    drift_ks_alpha: float = 0.01
    # Severity of injected covariate drift used to stress-test the detector.
    drift_severities: tuple = (0.0, 0.25, 0.5, 0.75, 1.0)

    # (5) Confidence-based orchestration: route low-confidence cases from the
    #     cheap INT8 head to the full-precision multi-view ensemble.
    orchestration_enabled: bool = True
    orchestration_conf_thresh: float = 0.85

    # (6) HITL rejection gating: abstain on high-entropy / low-margin cases and
    #     hand them to a gastroenterologist.
    hitl_enabled: bool = True
    hitl_target_coverage: float = 0.90    # auto-decide 90%, refer 10%
    hitl_entropy_thresh: float = 0.55
    hitl_margin_thresh: float = 0.20


# Six single-component knock-outs + the full system -> the 6-axis radar chart
# and ablation_study_results.csv.
ABLATION_VARIANTS = [
    "Full-IM",
    "w/o Privacy",
    "w/o TTA",
    "w/o Compression",
    "w/o DriftDetect",
    "w/o Orchestration",
    "w/o HITL",
    "No-IM (raw argmax)",   # reference: what every baseline gets by default
]


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class EvalCfg:
    spec_operating_point: float = 0.90   # sensitivity @ 90% specificity
    ece_bins: int = 10
    n_bootstrap: int = 2000              # for 95% CI on ROC curves
    alpha_significance: float = 0.05
    latency_warmup: int = 10
    latency_repeats: int = 50


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class FigCfg:
    style: str = "seaborn-v0_8-paper"
    dpi: int = 300
    save_pdf: bool = True
    save_png: bool = True
    font_family: str = "DejaVu Sans"
    base_fontsize: int = 9
    # Colour-blind-safe qualitative palette (Okabe-Ito derived).
    palette: tuple = ("#0072B2", "#E69F00", "#009E73", "#CC79A7",
                      "#56B4E9", "#D55E00", "#F0E442", "#999999")
    proposed_color: str = "#D55E00"
    baseline_color: str = "#0072B2"


@dataclass(frozen=True)
class Config:
    preproc: PreprocCfg = field(default_factory=PreprocCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    fed: FedCfg = field(default_factory=FedCfg)
    im: IMCfg = field(default_factory=IMCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)
    fig: FigCfg = field(default_factory=FigCfg)

    seeds: tuple = tuple(SEEDS)
    classes: tuple = tuple(CLASSES)
    methods: tuple = tuple(METHODS)
    modality: str = MODALITY

    # CPU-only box; cap threads so a runaway BLAS pool cannot thrash the
    # 15.6 GB / 5.2 GB-free machine.
    n_threads: int = min(10, os.cpu_count() or 4)
    device: str = "cpu"


CFG = Config()


# --------------------------------------------------------------------------
# CSV schemas - declared once, enforced by the writers in src/io_utils.py
# --------------------------------------------------------------------------
SCHEMA_MAIN = [
    "seed", "method_name", "modality", "auc_roc", "sensitivity_at_90spec",
    "specificity", "f1_score", "accuracy", "latency_ms", "throughput_fps",
    "model_size_mb", "ece",
]

SCHEMA_ABLATION = [
    "seed", "ablation_variant", "auc_roc", "f1_score", "latency_ms",
    "hitl_flag_rate", "ece", "privacy_budget_epsilon",
]

SCHEMA_SUBGROUP = [
    "seed", "method", "modality", "lesion_stage", "total_samples",
    "false_negatives", "false_positives", "auc_roc", "sensitivity",
]
