from torch.utils.data import DataLoader
import torchvision.datasets as datasets

from augmentation import get_transforms

USE_TRAIN_SUBSET_ONLY = True
_LAST_DATA_DIR = "./data"


def get_last_data_dir() -> str:
    return _LAST_DATA_DIR


def get_train_dataset_loader(
    data_dir,
    batch_size,
    generator_train,

):
    global _LAST_DATA_DIR
    _LAST_DATA_DIR = data_dir

    assert USE_TRAIN_SUBSET_ONLY, "USE_TRAIN_SUBSET_ONLY must be True"
    train_dataset = datasets.CIFAR100(
        root=data_dir,
        train=USE_TRAIN_SUBSET_ONLY,
        download=True,
        transform=get_transforms(train=True),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        generator=generator_train,
    )

    return train_dataset, train_loader
