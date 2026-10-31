import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, Subset
from torchvision.datasets import CIFAR10

from shared.protocol import DATA_SEED

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

def get_transform(augment: bool):
    steps = []
    if augment:
        steps += [
            transforms.RandomCrop(32, padding=4),   
            transforms.RandomHorizontalFlip(),     
        ]
    steps += [transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
    return transforms.Compose(steps)

def get_cifar10(train: bool = True, root: str = "./data", augment: bool = False) -> CIFAR10:

    if augment and not train:
        raise ValueError("augmentation is for training data only")
    return CIFAR10(root=root, train=train, download=True, transform=get_transform(augment))

def slice_data(num_samples: int, slice_id: int, num_slices: int,
               seed: int = DATA_SEED) -> list[int]:
    if num_slices < 1:
        raise ValueError(f"num_slices must be at least 1, got {num_slices}")
    if num_slices > num_samples:
        raise ValueError(f"num_slices ({num_slices}) can't exceed num_samples ({num_samples})")
    if not 0 <= slice_id < num_slices:
        raise ValueError(f"slice_id must be between 0 and {num_slices - 1}, got {slice_id}")
    g = torch.Generator()
    g.manual_seed(seed)
    shuffled_indices = torch.randperm(num_samples, generator=g)
    return shuffled_indices[slice_id::num_slices].tolist()

def get_slice(dataset: Dataset, slice_id: int, num_slices: int,
              seed: int = DATA_SEED) -> Subset:
    indices = slice_data(len(dataset), slice_id, num_slices, seed)
    return Subset(dataset, indices)

if __name__ == "__main__":
    train = get_cifar10(True)
    test = get_cifar10(False)
    print("train:", len(train), "test:", len(test))
    image, label = train[0]
    print("first image:", image.shape, "label:", label, f"({train.classes[label]})")
    part = get_slice(train, slice_id=0, num_slices=5)
    print("slice 0 of 5:", len(part), "images")
    print("first 5 indices of slice 0 of 5:", slice_data(len(train), 0, 5)[:5])