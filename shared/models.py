"""Lattice model library.

Every model a job can use is registered here by name. A job only sends the
name (e.g. "cnn_bn"); every machine builds the identical model from this file,
so no code ever travels over the network (DESIGN.md D7).

To add a model: write its nn.Module, then add a ModelSpec to LIBRARY.
"""
from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import nn

from shared.model import SmallCNN


class BNCNN(nn.Module):
    """Three blocks of two 3x3 convs with batch norm, then global average pooling.

    Shapes: 3x32x32 -> 32x16x16 -> 64x8x8 -> 128x4x4 -> 128 -> 10.
    About 289k parameters (~1.2 MB): smaller than SmallCNN to upload, but
    roughly 3-4x more computation per image, because the convs see more pixels.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        def block(c_in: int, c_out: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # 128x4x4 -> 128x1x1: one average per channel
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


@dataclass(frozen=True)
class ModelSpec:
    name: str                                   # what jobs refer to, e.g. "cnn_bn"
    title: str                                  # shown on the website
    description: str
    build: Optional[Callable[[], nn.Module]]    # None for "coming soon" entries
    augment: bool                               # random crops/flips on training images
    batch_size: int
    lr: float
    momentum: float = 0.9
    weight_decay: float = 0.0

    @property
    def ready(self) -> bool:
        return self.build is not None

    def make_model(self) -> nn.Module:
        if not self.ready:
            raise ValueError(f"model '{self.name}' is not available yet")
        return self.build()

    def make_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        return torch.optim.SGD(
            model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay
        )

    def summary(self) -> dict:
        """JSON-friendly description for the API and the website's model picker."""
        params = sum(p.numel() for p in self.make_model().parameters()) if self.ready else None
        return {
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "parameters": params,
            "size_mb": round(params * 4 / 1e6, 1) if params else None,
            "ready": self.ready,
        }


LIBRARY: dict[str, ModelSpec] = {
    "cnn_small": ModelSpec(
        name="cnn_small",
        title="Small CNN",
        description="Fastest. 3 conv layers; plateaus around 75% on CIFAR-10.",
        build=SmallCNN,
        augment=False,
        batch_size=64,
        lr=0.01,
    ),
    "cnn_bn": ModelSpec(
        name="cnn_bn",
        title="CNN with batch norm",
        description="Recommended. 6 conv layers with batch norm and augmentation: "
                    "higher accuracy, slower rounds.",
        build=BNCNN,
        augment=True,
        batch_size=128,
        lr=0.05,
        weight_decay=5e-4,
    ),
    "resnet_finetune": ModelSpec(
        name="resnet_finetune",
        title="Pretrained ResNet (fine-tune)",
        description="Coming soon. Starts from a model pretrained on millions of images.",
        build=None,
        augment=True,
        batch_size=64,
        lr=0.01,
    ),
}

DEFAULT_MODEL = "cnn_bn"


def get_spec(name: str) -> ModelSpec:
    """Look up a model that can be used now; a clear error otherwise."""
    spec = LIBRARY.get(name)
    if spec is None:
        available = ", ".join(n for n, s in LIBRARY.items() if s.ready)
        raise ValueError(f"unknown model '{name}'. Available: {available}")
    if not spec.ready:
        raise ValueError(f"model '{name}' is not available yet")
    return spec


def list_models() -> list[dict]:
    return [spec.summary() for spec in LIBRARY.values()]