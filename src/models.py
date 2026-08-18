"""
Frozen ImageNet encoders + the federated adapter head.

Why a frozen trunk?
-------------------
The target deployment (and the only hardware available here) is a CPU-only
edge box: measured 10.2 img/s for end-to-end ResNet-18 training at 224 px,
which makes 7 methods x 5 seeds x 4 Dirichlet settings of end-to-end
federated training physically impossible (>90 h).  We therefore adopt
*parameter-efficient federated learning*: a publicly pre-trained encoder is
frozen and shared, and only a small adapter head is federated.

This is not a shortcut that favours the proposed method - it is applied
identically to every baseline, so all comparisons remain fair - and it is
independently desirable, because only the head is transmitted each round
(kilobytes instead of tens of megabytes), which is what a bandwidth-limited
hospital link can actually sustain.

Head topology
-------------
    fc1 (Linear) -> bn (BatchNorm1d) -> act (GELU) -> drop -> fc2 (Linear)

  * `bn`  is what TENT-style test-time adaptation updates (affine params only).
  * `fc2` is the client-private personalisation layer for FedPer / FedGIM.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .config import CFG

_BACKBONE_BUILDERS = {
    "resnet18": (torchvision.models.resnet18,
                 torchvision.models.ResNet18_Weights.IMAGENET1K_V1),
    "efficientnet_b0": (torchvision.models.efficientnet_b0,
                        torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1),
    "densenet121": (torchvision.models.densenet121,
                    torchvision.models.DenseNet121_Weights.IMAGENET1K_V1),
}


class FrozenEncoder(nn.Module):
    """Pre-trained CNN with the classifier stripped, permanently in eval mode.

    Exposes `target_layer` so Grad-CAM++ can hook the last convolutional
    block.  Gradients still flow through frozen weights, so CAM generation is
    fully valid even though no encoder parameter is ever updated.
    """

    def __init__(self, name: str, pretrained: bool = True):
        super().__init__()
        if name not in _BACKBONE_BUILDERS:
            raise ValueError(f"unknown backbone {name!r}")
        builder, weights = _BACKBONE_BUILDERS[name]
        net = builder(weights=weights if pretrained else None)
        self.name = name

        if name == "resnet18":
            self.features = nn.Sequential(*list(net.children())[:-2])
            self.target_layer = self.features[-1]          # layer4
            self.out_dim = 512
        elif name == "efficientnet_b0":
            self.features = net.features
            self.target_layer = self.features[-1]          # final ConvBNAct
            self.out_dim = 1280
        else:  # densenet121
            self.features = net.features
            self.target_layer = self.features.norm5
            self.out_dim = 1024

        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):        # never leaves eval mode
        return super().train(False)

    def forward_map(self, x: torch.Tensor) -> torch.Tensor:
        """Spatial feature map (B, C, H', W') - the Grad-CAM++ target."""
        fm = self.features(x)
        if self.name == "densenet121":
            fm = F.relu(fm, inplace=False)
        return fm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pooled penultimate embedding (B, out_dim)."""
        return torch.flatten(F.adaptive_avg_pool2d(self.forward_map(x), 1), 1)


class MultiEncoder(nn.Module):
    """Concatenates several frozen encoders into one embedding.

    Combining a residual (ResNet-18) and an inverted-residual / squeeze-excite
    (EfficientNet-B0) trunk gives complementary texture and shape evidence;
    the benefit is quantified in the backbone ablation.
    """

    def __init__(self, names=None, pretrained: bool = True):
        super().__init__()
        names = list(names or CFG.model.backbones)
        self.encoders = nn.ModuleList(
            [FrozenEncoder(n, pretrained) for n in names])
        self.names = names
        self.out_dim = sum(e.out_dim for e in self.encoders)

    def train(self, mode: bool = True):
        return super().train(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([e(x) for e in self.encoders], dim=1)

    def slice_of(self, name: str) -> slice:
        """Column range this encoder occupies in the concatenated vector."""
        start = 0
        for e in self.encoders:
            if e.name == name:
                return slice(start, start + e.out_dim)
            start += e.out_dim
        raise KeyError(name)


class AdapterHead(nn.Module):
    """The only federated component. ~0.5 M params -> ~2 MB on the wire."""

    def __init__(self, in_dim: int, n_classes: int,
                 hidden: int | None = None, dropout: float | None = None):
        super().__init__()
        hidden = hidden or CFG.model.hidden_dim
        dropout = CFG.model.dropout if dropout is None else dropout
        self.fc1 = nn.Linear(in_dim, hidden)
        self.bn = nn.BatchNorm1d(hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, n_classes)
        self.in_dim, self.n_classes = in_dim, n_classes

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.drop(self.act(self.bn(self.fc1(z))))
        return self.fc2(h)

    def embed(self, z: torch.Tensor) -> torch.Tensor:
        """Penultimate representation - used by the drift detector."""
        return self.act(self.bn(self.fc1(z)))


class GIEDNet(nn.Module):
    """encoder + head, end-to-end from pixels. Used for XAI and latency."""

    def __init__(self, encoder: MultiEncoder, head: AdapterHead):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
class ClassBalancedLoss(nn.Module):
    """Cui et al. (CVPR'19) effective-number re-weighting.

    GIED is 8.9:1 imbalanced and the rarest class (Cancer, 402 images) is the
    one that must never be missed, so FedGIM trains with this rather than
    plain cross-entropy.
    """

    def __init__(self, samples_per_class, beta: float = 0.999,
                 n_classes: int | None = None):
        super().__init__()
        counts = torch.as_tensor(samples_per_class, dtype=torch.float32)
        counts = torch.clamp(counts, min=1.0)
        eff = 1.0 - torch.pow(beta, counts)
        w = (1.0 - beta) / eff
        w = w / w.sum() * (n_classes or len(counts))
        self.register_buffer("weight", w)

    def forward(self, logits, target):
        return F.cross_entropy(logits, target, weight=self.weight.to(logits.device))


def make_head(in_dim: int, n_classes: int) -> AdapterHead:
    return AdapterHead(in_dim, n_classes)


def clone_state(sd: dict) -> dict:
    return {k: v.detach().clone() for k, v in sd.items()}
