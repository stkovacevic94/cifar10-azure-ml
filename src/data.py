from albumentations.augmentations.transforms import HorizontalFlip, RandomBrightness, VerticalFlip
import pytorch_lightning as plt

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import datasets
from torch.utils.data import random_split

import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2

import argparse

class CIFAR10(Dataset):
    def __init__(self, root, train, download=False, transform=None):
        self.dataset = datasets.CIFAR10(root=root, train=train, download=download)
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.transform is not None:
            augmented = self.transform(image=np.array(image))
            image = augmented['image']
        return image, label

class CIFAR10DataModule(plt.LightningDataModule):
    def __init__(self, data_path, batch_size):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.dims = (3, 32, 32)

        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(),
            ToTensorV2()
        ])

    def prepare_data(self):
        # Download if failed to fetch from Azure Blob Storage
        CIFAR10(root=self.data_path, train=True, download=True)
        CIFAR10(root=self.data_path, train=False, download=True)
    
    def setup(self, stage:str=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            train_dataset_full = CIFAR10(self.data_path, train=True, transform=self.transform)
            self.train_dataset, self.valid_dataset = random_split(train_dataset_full, [45000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.test_dataset = CIFAR10(self.data_path, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers=4, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    @staticmethod
    def add_data_specific_args(parent_parser: argparse.ArgumentParser):
        parser = parent_parser.add_argument_group("Data")
        parser.add_argument('--data_path', type=str, required=True, help='Dataset path')
        parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
        return parent_parser