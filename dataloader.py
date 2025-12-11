import numpy as np
from typing import Tuple, Optional, List
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import yaml
from dataset import SmartPixelsDataset


def create_dataloaders(
    config_path: str,
    batch_size: int = 128,
    shuffle: bool = True,
    transform: callable = None,
    target_transform: callable = None,
    val_size: float = 0.2,
) -> Tuple[DataLoader, DataLoader]:
    
    config_dict = yaml.safe_load(open(config_path, "r"))

    train_image_paths = [config_dict['root'] + t[0] for t in config_dict['train']]
    train_label_paths = [config_dict['root'] + t[1] for t in config_dict['train']]

    test_image_paths = [config_dict['root'] + t[0] for t in config_dict['test']]
    test_label_paths = [config_dict['root'] + t[1] for t in config_dict['test']]

    # Train-test split of indices
    if val_size > 0.0:
        train_image_paths, val_image_paths, train_label_paths, val_label_paths = train_test_split(
            train_image_paths, train_label_paths,
            test_size=val_size,
            random_state=42,
        )
    else:
        val_image_paths = []
        val_label_paths = []

    # Create datasets
    train_dataset = SmartPixelsDataset(
        train_image_paths,
        train_label_paths,
        shuffle=shuffle,
        transform=transform,
        target_transform=target_transform
    )

    val_dataset = SmartPixelsDataset(
        val_image_paths,
        val_label_paths,
        shuffle=shuffle,
        transform=transform,
        target_transform=target_transform
    )

    test_dataset = SmartPixelsDataset(
        test_image_paths,
        test_label_paths,
        shuffle=shuffle,
        transform=transform,
        target_transform=target_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Example usage

    train_loader, val_loader, test_loader = create_dataloaders(
        config_path="config/v3.yml",
        batch_size=64,
        shuffle=True,
        transform=lambda x: x.reshape((20, 13, 21)),
        target_transform=None,
        val_size=0.2
    )

    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    print(f"Number of test batches: {len(test_loader)}")


    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx}: images shape = {images.shape}, labels shape = {labels.shape}")
        if batch_idx == 2:  # Just show first 3 batches
            break