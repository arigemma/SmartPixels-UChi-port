import numpy as np
from typing import Tuple, Optional, List
from sklearn.discriminant_analysis import StandardScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import yaml
import joblib
from torchvision.transforms import Compose, Lambda
from utils import SymLogTransform
import torch
from glob import glob
from dataset import SmartPixelsDataset


def create_dataloaders(
    config_path: str,
    batch_size: int = 128,
    shuffle: bool = True,
    target_type: str = "regression",  # "regression" or "classification"
    val_size: float = 0.2,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    
    config_dict = yaml.safe_load(open(config_path, "r"))

    train_image_paths = [glob(config_dict['root'] + t[0]) for t in config_dict['train']]
    train_label_paths = [glob(config_dict['root'] + t[1]) for t in config_dict['train']]

    test_image_paths = [glob(config_dict['root'] + t[0]) for t in config_dict['test']]
    test_label_paths = [glob(config_dict['root'] + t[1]) for t in config_dict['test']]

    print(f"train_image_paths: {train_image_paths[:3]} ...")
    print(f"train_label_paths: {train_label_paths[:3]} ...")
    print(f"test_image_paths: {test_image_paths[:3]} ...")
    print(f"test_label_paths: {test_label_paths[:3]} ...")

    scaler_paths = config_dict['scalers']
    scaler_X = joblib.load(scaler_paths[0])
    scaler_y = joblib.load(scaler_paths[1])

    transform = Compose([
        SymLogTransform(),
        Lambda(lambda x: scaler_X.transform(x.reshape(-1, 1)) if scaler_X else x),
        Lambda(lambda x: x.reshape(1, 20, 13, 21)),
        Lambda(lambda x: torch.tensor(x, dtype=torch.float32))
    ])

    if target_type == "regression":
        target_transform = Compose([
            Lambda(lambda y: scaler_y.transform(y.reshape(1, -1)).flatten() if scaler_y else y),
            Lambda(lambda y: torch.tensor(y, dtype=torch.float32))
        ])
    elif target_type == "classification":
        def get_label(y):
            label = torch.zeros(3)
            if 0 <= y[8] <= 0.2:  # positively charged and below 200 MeV
                label[0] = 1
            elif -0.2 <= y[8] < 0.:  # negatively charged and below 200 MeV
                label[1] = 1
            else:
                label[2] = 1  # above 200 MeV
            return label
        target_transform = Lambda(get_label)
    else:
        raise ValueError(f"Unknown target_type: {target_type}")

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
    ds_kwargs = {
        "shuffle": shuffle,
        "transform": transform,
        "target_transform": target_transform
    }

    train_dataset = SmartPixelsDataset(
        train_image_paths, train_label_paths, **ds_kwargs
    )
    val_dataset = SmartPixelsDataset(
        val_image_paths, val_label_paths, **ds_kwargs
    )
    test_dataset = SmartPixelsDataset(
        test_image_paths, test_label_paths, **ds_kwargs
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": 0,
        "pin_memory": True
    }

    train_loader = DataLoader(train_dataset, **loader_kwargs)
    val_loader = DataLoader(val_dataset, **loader_kwargs)
    test_loader = DataLoader(test_dataset, **loader_kwargs)

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
