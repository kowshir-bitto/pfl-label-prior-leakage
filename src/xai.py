"""
Grad-CAM++ implemented from first principles (no third-party CAM library).

Grad-CAM++ (Chattopadhay et al., WACV'18) replaces Grad-CAM's global-average
gradient pooling with a pixel-wise weighting

    a_ij^kc = (d2Y^c/dA_ij^k 2) /
              (2 d2Y^c/dA_ij^k 2 + sum_ab A_ab^k d3Y^c/dA_ij^k 3)

Writing Y^c = exp(S^c) makes the higher derivatives collapse to powers of the
first-order gradient of the score, which is all autograd needs to supply:

    a_ij = g_ij^2 / (2 g_ij^2 + (sum_ab A_ab) g_ij^3)
    w_k  = sum_ij a_ij * relu(g_ij)
    L^c  = relu( sum_k w_k A^k )

Because the encoder is frozen rather than absent, gradients still propagate
through it normally - freezing only stops the optimiser, not autograd - so CAM
extraction is exactly as valid as for an end-to-end fine-tuned network.

NOTE ON GROUND TRUTH: the GIED release ships image-level labels only; it
contains no lesion masks or bounding boxes.  We therefore never draw a
"ground-truth boundary".  Qualitative panels compare baseline and proposed
attributions against each other and report quantitative localisation proxies
(energy concentration, CAM entropy, inter-model agreement) instead.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .config import CFG


class GradCAMPlusPlus:
    """Hook-based Grad-CAM++ for any (encoder -> head) model.

    Parameters
    ----------
    model
        Callable module mapping an image batch to class logits.
    target_layer
        The convolutional module whose output activations are explained
        (`layer4` for ResNet, the final ConvBNAct for EfficientNet,
        `features.norm5` for DenseNet).
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model.eval()
        self.target_layer = target_layer
        self._acts: torch.Tensor | None = None
        self._grads: torch.Tensor | None = None
        self._h = [
            target_layer.register_forward_hook(self._save_act),
            target_layer.register_full_backward_hook(self._save_grad),
        ]

    def _save_act(self, _m, _i, out):
        self._acts = out

    def _save_grad(self, _m, _gi, gout):
        self._grads = gout[0]

    def remove(self):
        for h in self._h:
            h.remove()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()

    def __call__(self, x: torch.Tensor, class_idx=None,
                 out_size: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return (cams (B,H,W) in [0,1], predicted/target class indices)."""
        x = x.clone().requires_grad_(True)
        logits = self.model(x)
        if class_idx is None:
            targets = logits.argmax(dim=1)
        else:
            targets = (torch.as_tensor(class_idx).view(-1)
                       .expand(logits.shape[0]).to(logits.device))

        score = logits.gather(1, targets.view(-1, 1)).sum()
        self.model.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad = None
        score.backward(retain_graph=False)

        A = self._acts.detach()                     # (B, K, H, W)
        g = self._grads.detach()                    # (B, K, H, W)
        g2, g3 = g ** 2, g ** 3
        sum_A = A.sum(dim=(2, 3), keepdim=True)
        denom = 2.0 * g2 + sum_A * g3
        alpha = torch.where(denom.abs() > 1e-9, g2 / denom,
                            torch.zeros_like(denom))
        weights = (alpha * F.relu(g)).sum(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * A).sum(dim=1))      # (B, H, W)

        size = out_size or CFG.preproc.img_size
        cam = F.interpolate(cam.unsqueeze(1), size=(size, size),
                            mode="bilinear", align_corners=False).squeeze(1)
        cam = cam.cpu().numpy()
        mn = cam.min(axis=(1, 2), keepdims=True)
        mx = cam.max(axis=(1, 2), keepdims=True)
        cam = (cam - mn) / np.maximum(mx - mn, 1e-8)
        return cam, targets.cpu().numpy()


# ---------------------------------------------------------------------------
# Quantitative attribution descriptors (no masks required)
# ---------------------------------------------------------------------------
def cam_energy_concentration(cam: np.ndarray, top_frac: float = 0.10) -> float:
    """Share of total CAM mass inside its hottest `top_frac` of pixels.

    A focal, lesion-shaped explanation concentrates mass; a diffuse or
    background-driven one spreads it.  Higher is better.
    """
    flat = np.sort(cam.ravel())[::-1]
    k = max(1, int(len(flat) * top_frac))
    tot = flat.sum()
    return float(flat[:k].sum() / tot) if tot > 0 else 0.0


def cam_entropy(cam: np.ndarray) -> float:
    """Normalised Shannon entropy of the CAM treated as a distribution."""
    p = cam.ravel().astype(np.float64)
    s = p.sum()
    if s <= 0:
        return 1.0
    p = p / s
    nz = p[p > 0]
    return float(-(nz * np.log(nz)).sum() / np.log(len(p)))


def cam_agreement(a: np.ndarray, b: np.ndarray, thresh: float = 0.5) -> float:
    """IoU between the two CAMs' above-threshold supports."""
    ma, mb = a >= thresh, b >= thresh
    union = np.logical_or(ma, mb).sum()
    return float(np.logical_and(ma, mb).sum() / union) if union else 0.0


def cam_border_mass(cam: np.ndarray, frac: float = 0.15) -> float:
    """Fraction of CAM mass in the image border ring.

    Endoscopic frames are vignetted; attribution that leaks into the dark
    border is spurious.  Lower is better.
    """
    h, w = cam.shape
    b = max(1, int(min(h, w) * frac))
    mask = np.ones_like(cam, dtype=bool)
    mask[b:h - b, b:w - b] = False
    tot = cam.sum()
    return float(cam[mask].sum() / tot) if tot > 0 else 0.0


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def overlay_cam(img_bgr: np.ndarray, cam: np.ndarray,
                alpha: float = 0.42) -> np.ndarray:
    """Blend a JET heat map over the image; returns RGB uint8."""
    h, w = img_bgr.shape[:2]
    c = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
    hm = cv2.applyColorMap(np.uint8(255 * np.clip(c, 0, 1)), cv2.COLORMAP_JET)
    out = cv2.addWeighted(hm, alpha, img_bgr, 1 - alpha, 0)
    return out[:, :, ::-1]


def cam_contour(img_bgr: np.ndarray, cam: np.ndarray, thresh: float = 0.5,
                color=(0, 255, 255), width: int = 2) -> np.ndarray:
    """Draw the CAM's iso-contour - a boundary-style rendering of the region
    the model considers evidential (NOT a ground-truth annotation)."""
    h, w = img_bgr.shape[:2]
    c = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
    mask = (c >= thresh).astype(np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = img_bgr.copy()
    cv2.drawContours(out, cnts, -1, color, width)
    return out[:, :, ::-1]


def to_tensor(img_bgr: np.ndarray) -> torch.Tensor:
    """BGR uint8 HWC -> normalised RGB CHW float tensor."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t = torch.from_numpy(rgb).permute(2, 0, 1)
    mean = torch.tensor(CFG.model.mean).view(3, 1, 1)
    std = torch.tensor(CFG.model.std).view(3, 1, 1)
    return (t - mean) / std
