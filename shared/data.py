import torch
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import Subset
from shared.protocol import DATA_SEED

CIFAR10_MEAN=(0.4914, 0.4822, 0.4465)
CIFAR10_STD=(0.2470, 0.2435, 0.2616)

def get_cifar10(train: bool = True, root: str = "./data") -> CIFAR10:
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN,CIFAR10_STD)
    ])

    dataset=CIFAR10(
        root=root,
        train=train,
        download=True,
        transform=transform
    )
    
    return dataset

def slice_data(no_of_samples:int,slice_id:int,num_slices:int,seed: int = DATA_SEED) -> list[int]:
    if num_slices < 1:
        raise ValueError(f"num_slices must be at least 1, got {num_slices}")
    if not 0 <= slice_id < num_slices:
        raise ValueError(
            f"slice_id must be between 0 and {num_slices - 1}, got {slice_id}"
        )
    if num_slices > no_of_samples:
        raise ValueError(
            f"num_slices ({num_slices}) can't exceed num_samples ({no_of_samples})"
        )
    g=torch.Generator()
    g.manual_seed(seed)
    shuffled_indices=torch.randperm(no_of_samples,generator=g)
    final_indices=shuffled_indices[slice_id::num_slices].tolist()
    return final_indices

def get_slice(dataset, slice_id: int, num_slices: int, seed: int = DATA_SEED) -> Subset:
    indices = slice_data(len(dataset), slice_id, num_slices, seed)
    return Subset(dataset, indices)
    
if __name__=="__main__":
    train=get_cifar10(True)    
    test=get_cifar10(False)
    print("train:", len(train), "test:", len(test))
    
    image, label = train[0]
    print(image.shape, label)
    
    part = get_slice(train, slice_id=0, num_slices=5)
    image, label = part[0]
    print("slice 0 of 5:", len(part), "images; first:", image.shape, label)

    print("first 5 indices of slice 0 of 5:", slice_data(len(train), 0, 5)[:5])