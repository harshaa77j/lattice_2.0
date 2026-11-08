import io

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from shared.data import get_transform

IMAGE_SIZE = 32


class ArrayDataset(Dataset):
    """Images held in memory as raw pixels; transformed (and augmented, if asked) on access."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, augment: bool = False):
        self.images, self.labels = images, labels
        self.transform = get_transform(augment)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.transform(Image.fromarray(self.images[i])), int(self.labels[i])


def pack_arrays(images: np.ndarray, labels: np.ndarray) -> bytes:
    """Arrays -> compressed .npz bytes (for files on disk and for sending to volunteers)."""
    buf = io.BytesIO()
    np.savez_compressed(buf, images=images.astype(np.uint8), labels=labels.astype(np.uint16))
    return buf.getvalue()


def unpack_arrays(data: bytes):
    """Compressed .npz bytes -> (images, labels). Pickle stays disabled."""
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        return z["images"], z["labels"].astype(np.int64)