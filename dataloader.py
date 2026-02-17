import numpy as np
import yaml
from typing import Tuple, Optional, List, Callable, Any
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import Identity
from utils import SymLogTransform
from dataset import SmartPixelsDataset, UnderSampler


N_TIME = 20
N_Y = 13
N_X = 21
LAST_FRAME = N_TIME - 1
N_SLICES = 8


def _y_profile_last_frame(x: torch.Tensor) -> torch.Tensor:
    """Mean over x-dimension for last time frame."""
    return x[LAST_FRAME].mean(axis=-1)

def _y_profile_over_time(x: torch.Tensor) -> torch.Tensor:
    """Mean over x-dimension for all time steps."""
    return x.mean(axis=-1)

def _get_scale_function(mean, std):
    # avoid division by zero
    std = np.where(np.isclose(std, 0.0), 1.0, std)
    return lambda x: (x - mean) / std


def get_transform(
    input_type: str = "full",  # "full", "last-frame", "y-size", "y-profile", "y-profile-timing"
    scale_dict: Optional[dict] = None,
) -> Callable[[Any, Any], torch.Tensor]:
    """
    Build a transform that consumes (x, y) but modifies only x.
    """

    if scale_dict is not None:
        scale_x = _get_scale_function(scale_dict["mean_X"], scale_dict["std_X"])
        scale_y = _get_scale_function(scale_dict["mean_y"], scale_dict["std_y"])
        scaling_enabled = True
    else:
        scale_x = Identity()
        scale_y = Identity()
        scaling_enabled = False

    def transform(raw_x, raw_y) -> torch.Tensor:
        # --- preprocess x ---
        x = SymLogTransform()(raw_x)
        x = scale_x(x)
        x = x.reshape(N_TIME, N_Y, N_X)

        # --- preprocess y ---
        y_scaled = scale_y(raw_y)
        y0 = y_scaled[7]  # eigth feature

        # --- build features ---
        if input_type == "full":
            features = x

        elif input_type == "last-frame":
            features = x[LAST_FRAME]

        elif input_type == "y-size":
            # use unscaled raw_x to count nonzero entries
            raw_x_reshaped = raw_x.reshape(N_TIME, N_Y, N_X)
            y_profile = raw_x_reshaped[LAST_FRAME].mean(axis=-1)
            cluster_size = (y_profile > 0).sum()
            features = np.array([y0, cluster_size])

        elif input_type == "y-profile":
            y_profile = _y_profile_last_frame(x)
            features = np.concatenate(([y0], y_profile))

        elif input_type == "y-profile-timing":
            y_profile = _y_profile_over_time(x)
            slices = y_profile[:N_SLICES].T  # (13, 8)
            features = np.concatenate(([y0], slices.flatten()))  # (1 + 13*8,)

        else:
            raise ValueError(f"Unknown input_type: {input_type}")

        return torch.as_tensor(features, dtype=torch.float32)

    print(
        f"Built transform with input_type={input_type}, "
        f"scaling={'enabled' if scaling_enabled else 'disabled'}"
    )

    return transform


def get_target_transform(
    target_type: str = "regression",  # "regression" or "classification"
    scale_dict: Optional[dict] = None,
) -> Callable[[Any], torch.Tensor]:
    """
    The transform that builds the targets.
    """

    if scale_dict is not None:
        scale_y = _get_scale_function(scale_dict["mean_y"], scale_dict["std_y"])
    else:
        scale_y = Identity()

    def target_transform(y):
        if target_type == "classification":
            # extract pt (9th feature)
            pt = y[..., 8]  # preserve batch dimension if present.

            label = np.zeros((*pt.shape, 3))  # Shape: (3,) for single sample, (batch_size, 3) for batch

            label[..., 0] = (0. <= pt) & (pt <= 0.2)
            label[..., 1] = (-0.2 <= pt) & (pt < 0.)
            label[..., 2] = (pt < -0.2) | (pt > 0.2)
            
            y = label
        elif target_type == "classification-2-GeV":
            # extract pt (9th feature)
            pt = y[..., 8]  # preserve batch dimension if present.

            label = np.zeros((*pt.shape, 3))  # Shape: (3,) for single sample, (batch_size, 3) for batch

            label[..., 0] = (0. <= pt) & (pt <= 2.)
            label[..., 1] = (-2. <= pt) & (pt < 0.)
            label[..., 2] = (pt < -2.) | (pt > 2.)

            y = label
        elif target_type == "regression":
            y = scale_y(y)
        elif target_type == "raw":
            y = y
        else:
            raise ValueError(f"Unknown target_type: {target_type}")
        y = torch.as_tensor(y, dtype=torch.float32)            
        return y

    print(f"Built target transform with target_type={target_type}, scaling={'enabled' if scale_dict else 'disabled'}")

    return target_transform


def create_dataloaders(
    config_path: str,
    batch_size: int = 128,
    shuffle: bool = True,
    input_type: str = "full",  # "full", "last-frame", "y-size", "y-profile", "y-profile-timing"
    target_type: str = "regression",  # "regression" or "classification"
    balance_classes: bool = False,
    val_size: float = 0.2,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    
    config_dict = yaml.safe_load(open(config_path, "r"))

    train_image_paths = [config_dict['root'] + t[0] for t in config_dict['train']]
    train_label_paths = [config_dict['root'] + t[1] for t in config_dict['train']]

    test_image_paths = [config_dict['root'] + t[0] for t in config_dict['test']]
    test_label_paths = [config_dict['root'] + t[1] for t in config_dict['test']]

    assert len(train_image_paths) > 0, "No training data found"
    assert len(train_image_paths) == len(train_label_paths), "Mismatch in number of training images and labels"
    assert len(test_image_paths) == len(test_label_paths), "Mismatch in number of test images and labels"

    # sym log on x in every case
    scale_dict = config_dict.get("scaling", None)
    transform = get_transform(input_type, scale_dict)
    target_transform = get_target_transform(target_type, scale_dict)

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
        "transform": transform,
        "target_transform": target_transform,
    }

    train_dataset = SmartPixelsDataset(
        train_image_paths, train_label_paths, shuffle=shuffle, **ds_kwargs
    )
    val_dataset = SmartPixelsDataset(
        val_image_paths, val_label_paths, shuffle=False, **ds_kwargs
    )
    test_dataset = SmartPixelsDataset(
        test_image_paths, test_label_paths, shuffle=False, **ds_kwargs
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": 0,
        # "pin_memory": True
    }

    # if balance_classes:
    #     sampler = UnderSampler(train_dataset, target_class_ratios={0: 1/3, 1: 1/3, 2: 1/3}, max_oversampling_rate=35.0)
    #     # sampler = UnderSampler(train_dataset, target_class_ratios={0: 1/6, 1: 1/6, 2: 2/3}, max_oversampling_rate=3.0) # collapses
    #     # sampler = UnderSampler(train_dataset, target_class_ratios={0: 1/4, 1: 1/4, 2: 1/2}, max_oversampling_rate=3.0) # collapses
    #     # sampler = UnderSampler(train_dataset, target_class_ratios={0: 0.3, 1: 0.3, 2: 0.4}, max_oversampling_rate=3.0) # works well
    #     # sampler = UnderSampler(train_dataset, target_class_ratios={0: 0.28, 1: 0.28, 2: 0.44}, max_oversampling_rate=3.0) # works well
    # else:
    #     sampler = None

    # train_loader = DataLoader(train_dataset, sampler=sampler, **loader_kwargs)
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
        input_type="full",
        target_type="regression",
        val_size=0.2
    )

    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    print(f"Number of test batches: {len(test_loader)}")

    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx}: images shape = {images.shape}, labels shape = {labels.shape}")
        if batch_idx == 2:  # Just show first 3 batches
            break
