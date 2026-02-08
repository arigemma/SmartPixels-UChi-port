import numpy as np
from typing import Tuple, Optional, List, Callable, Any
from sklearn.model_selection import train_test_split
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Lambda
from torch.nn import Identity
from utils import SymLogTransform
from dataset import SmartPixelsDataset, MySampler


def get_scale_function(mean, std):
    # avoid division by zero
    std = np.where(np.isclose(std, 0.0), 1.0, std)
    return lambda x: (x - mean) / std


def get_transform(
    input_type: str = "full",  # "full", "last-frame", "y-profile"
    scale_dict: Optional[dict] = None,
) -> Callable[[Any, Any], torch.Tensor]:
    """
    The transform that build the inputs. Unlike normal, this takes
    both x and y, but modifies only x. (Beacuse of special annotation cases.)
    """
    
    if scale_dict is not None:
        scale_x = get_scale_function(scale_dict["mean_X"], scale_dict["std_X"])
        scale_y = get_scale_function(scale_dict["mean_y"], scale_dict["std_y"])
    else:
        scale_x = Identity()
        scale_y = Identity()

    def transform(x, y):
        x = SymLogTransform()(x)
        x = scale_x(x)
        x = x.reshape(20, 13, 21)

        if input_type == "last-frame":
            x = x[19, :, :]
        elif input_type == "y-profile":
            x = x[19, :, :].mean(axis=-1)
            y = scale_y(y)
            # add y-0 as the first element of the y-profile
            x = np.concatenate(([y[1]], x))

        x = torch.as_tensor(x, dtype=torch.float32)

        return x
    
    print(f"Built transform with input_type={input_type}, scaling={'enabled' if scale_dict else 'disabled'}")    

    return transform


def get_target_transform(
    target_type: str = "regression",  # "regression" or "classification"
    scale_dict: Optional[dict] = None,
) -> Callable[[Any], torch.Tensor]:
    """
    The transform that builds the targets.
    """

    if scale_dict is not None:
        scale_y = get_scale_function(scale_dict["mean_y"], scale_dict["std_y"])
    else:
        scale_y = Identity()

    def target_transform(y):
        if target_type == "classification":
            # extract pt (9th feature)
            pt = y[..., 8]  # preserve batch dimension if present.

            label = np.zeros((*pt.shape, 3))  # Shape: (3,) for single sample, (batch_size, 3) for batch

            label[..., 0] = (0 <= pt) & (pt <= 0.2)
            label[..., 1] = (-0.2 <= pt) & (pt < 0.)
            label[..., 2] = (pt < -0.2) | (pt > 0.2)
            
            y = label
        elif target_type == "regression":
            y = scale_y(y)
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
    input_type: str = "full",  # "full", "last-frame", "y-profile"
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

    if balance_classes:
        sampler = MySampler(train_dataset, target_class_ratios={0: 0.33, 1: 0.33, 2: 0.34})
    else:
        sampler = None

    train_loader = DataLoader(train_dataset, sampler=sampler, **loader_kwargs)
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
