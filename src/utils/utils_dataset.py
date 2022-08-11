import numpy as np
from torchvision.transforms import transforms
from pathlib import Path
from typing import List

from dataset_app.datasets import FashionMNIST_truncated
DATA_ROOT = Path("./data")
import numpy as np
from torchvision.transforms import transforms
from pathlib import Path
from typing import List

from dataset_app.datasets import (
    FashionMNIST_truncated,
    CIFAR10_truncated,
)
DATA_ROOT = Path("./data")

def load_dataset(dataset_name: str, train: bool = True, dataidxs: List[np.ndarray]=None,):
    if dataset_name == "FashionMNIST":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))])
        dataset = FashionMNIST_truncated(
            root=DATA_ROOT,
            dataidxs=dataidxs,
            train=train,
            transform=transform,
            download=True) 
    elif dataset_name == "CIFAR10":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))])
        root = DATA_ROOT / "CIFAR10" / "raw"
        dataset = CIFAR10_truncated(
            root=root,
            dataidxs=dataidxs,
            train=train,
            transform=transform,
        )
    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not Implemented")
    return dataset