from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import nn
import torch.nn.functional as F

from shared.model import SmallCNN

import math

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
    
class BasicBlock(nn.Module):
    """Two 3x3 convs plus a shortcut: the block learns a *change* to its input, which makes deep nets trainable."""

    def __init__(self, c_in: int, c_out: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.shortcut = nn.Sequential()
        if stride != 1 or c_in != c_out:     # match the shape when the block shrinks or widens
            self.shortcut = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False), nn.BatchNorm2d(c_out))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet20(nn.Module):
    """ResNet-20 for 32x32 images (He et al., 2016): three stages of three blocks, 16/32/64 channels.

    About 0.27M parameters (~1.1 MB) and roughly the same computation per image as BNCNN,
    but deeper, so it reaches higher accuracy.
    """

    def __init__(self, num_classes: int = 10, blocks_per_stage: int = 3):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True))
        layers, c_in = [], 16
        for c_out, stride in ((16, 1), (32, 2), (64, 2)):
            for i in range(blocks_per_stage):
                layers.append(BasicBlock(c_in, c_out, stride if i == 0 else 1))
                c_in = c_out
        self.stages = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, num_classes))

    def forward(self, x):
        return self.head(self.stages(self.stem(x)))

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
    schedule: str = "constant"  

    @property
    def ready(self) -> bool:
        return self.build is not None

    def make_model(self, num_classes: int = 10) -> nn.Module:
        if not self.ready:
            raise ValueError(f"model '{self.name}' is not available yet")
        return self.build(num_classes=num_classes)

    def lr_at(self, progress: float) -> float:
        """Learning rate at a point in training: progress 0.0 = start, 1.0 = end.

        "cosine" starts at lr and glides down along half a cosine wave to 2% of it,
        big steps early to learn fast, small steps late to settle.
        """
        if self.schedule != "cosine":
            return self.lr
        p = min(max(progress, 0.0), 1.0)
        return self.lr * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * p)))

    def make_optimizer(self, model: nn.Module, progress: float = 0.0) -> torch.optim.Optimizer:
        return torch.optim.SGD(
            model.parameters(), lr=self.lr_at(progress), momentum=self.momentum,
            weight_decay=self.weight_decay,
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
        schedule="cosine",
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
    "resnet20": ModelSpec(
        name="resnet20",
        title="ResNet-20",
        description="Highest accuracy. A 20-layer residual network with augmentation; "
                    "about the same cost per round as the batch-norm CNN.",
        build=ResNet20,
        augment=True,
        batch_size=128,
        lr=0.05,
        weight_decay=5e-4,
        schedule="cosine",
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