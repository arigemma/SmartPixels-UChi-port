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
ADC_THRESHOLDS = {
    # Three thresholds produce the four ADC codes used by the original
    # QKeras Model-2 ``adc2a`` operating points.
    "adc2a": [400.0, 1600.0, 2400.0],
}


class MaskFeaturesDataset(Dataset):
    def __init__(self, base_ds: Dataset, keep_mask: torch.Tensor):
        self.base_ds = base_ds
        self.keep_mask = keep_mask.detach().cpu().bool()

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        x, y = self.base_ds[idx]
        x = x.reshape(-1)
        x = x[self.keep_mask]
        return x, y

    def __getattr__(self, name):
        return getattr(self.base_ds, name)


def compute_keep_mask_from_loader(
    loader: DataLoader,
    thr: float = 0.95,
    num_batches: int = 10,
) -> torch.Tensor:
    mode_match_total = None
    total_n = 0

    for i, (xb, _) in enumerate(loader):
        if i >= num_batches:
            break

        xb = xb.reshape(xb.size(0), -1) 
        B, D = xb.shape

        mode_vals, _ = torch.mode(xb, dim=0)  
        mode_counts = (xb == mode_vals.unsqueeze(0)).sum(dim=0)  

        if mode_match_total is None:
            mode_match_total = mode_counts.clone()
        else:
            mode_match_total += mode_counts

        total_n += B

    mode_frac = mode_match_total.float() / float(total_n)
    keep_mask = mode_frac < thr
    return keep_mask

def _y_profile_last_frame(x: torch.Tensor) -> torch.Tensor:
    """Mean over x-dimension for last time frame."""
    return x[LAST_FRAME].mean(axis=-1)

def _y_profile_over_time(x: torch.Tensor) -> torch.Tensor:
    """Mean over x-dimension for all time steps."""
    return x.mean(axis=-1)


def _adc_y_profile(raw_x: np.ndarray, mapping: str) -> np.ndarray:
    """Apply an ADC mapping per pixel and sum the last-frame codes by row."""
    if mapping not in ADC_THRESHOLDS:
        raise ValueError(
            f"Unknown ADC mapping '{mapping}'. "
            f"Available mappings: {sorted(ADC_THRESHOLDS)}"
        )
    raw_x_reshaped = np.asarray(raw_x).reshape(N_TIME, N_Y, N_X)
    last_frame = raw_x_reshaped[LAST_FRAME]
    codes = np.digitize(last_frame, ADC_THRESHOLDS[mapping], right=False)
    return codes.sum(axis=-1).astype(np.float32)


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
        x = SymLogTransform()(raw_x)
        x = scale_x(x)
        x = x.reshape(N_TIME, N_Y, N_X)

        y_scaled = scale_y(raw_y)
        y0 = y_scaled[7]  # y-local

        if input_type == "full":
            features = x

        elif input_type == "last-frame":
            features = x[LAST_FRAME]

        elif input_type == "y-size":
            raw_x_reshaped = raw_x.reshape(N_TIME, N_Y, N_X)
            y_profile = raw_x_reshaped[LAST_FRAME].mean(axis=-1)
            cluster_size = (y_profile > 0).sum()
            features = np.array([y0, cluster_size])

        elif input_type == "y-profile":
            y_profile = _y_profile_last_frame(x)
            features = np.concatenate(([y0], y_profile))

        elif input_type.startswith("y-profile-adc"):
            adc_mapping = input_type[len("y-profile-"):]
            y_profile = scale_x(SymLogTransform()(_adc_y_profile(raw_x, adc_mapping)))
            features = np.concatenate(([y0], y_profile))

        elif input_type == "y-profile-timing":
            y_profile = _y_profile_over_time(x)          # (20, 13)
            slices = y_profile[:8].T                     # (13, 8)
            features = np.concatenate(([y0], slices.flatten()))

        elif input_type == "y-profile-timing-20":
            y_profile = _y_profile_over_time(x)          # (20, 13)
            slices = y_profile[:20].T                    # (13, 20)
            features = np.concatenate(([y0], slices.flatten()))   # (1 + 13*20,)

        elif input_type == "y-profile-timing-20-raw":
            y_profile = _y_profile_over_time(x)          # (20, 13)
            slices = y_profile[:20].T                    # (13, 20)
            features = slices.flatten()                  # (13*20,)

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
    label_format: str = "one-hot", # "index"
) -> Callable[[Any], torch.Tensor]:
    """
    The transform that builds the targets.
    """

    if scale_dict is not None:
        scale_y = _get_scale_function(scale_dict["mean_y"], scale_dict["std_y"])
    else:
        scale_y = Identity()

    def target_transform(y):
        if target_type == "raw":
            return torch.as_tensor(y, dtype=torch.float32)

        elif target_type == "regression":
            y = scale_y(y)
            return torch.as_tensor(y, dtype=torch.float32)

        elif target_type in ("classification", "classification-2-GeV"):
            boundary = 0.2 if target_type == "classification" else 2.0
            pt = y[..., 8]  # preserve batch dimension if present.

            c0 = (0 <= pt) & (pt <= boundary)
            c1 = (-boundary <= pt) & (pt < 0.0)
            c2 = (pt < -boundary) | (pt > boundary)

            idx = np.empty(pt.shape, dtype=np.int64)
            idx[c0] = 0
            idx[c1] = 1
            idx[c2] = 2
            index = torch.as_tensor(idx, dtype=torch.long)

            if label_format == "index":
                return index

            elif label_format == "one-hot":
                return torch.nn.functional.one_hot(index, num_classes=3).float()

            else:
                raise ValueError(f"Unknown label_format: {label_format}")
        else:
            raise ValueError(f"Unknown target_type: {target_type}")

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
    drop_dominant_features: bool = False,
    dominant_thr: float = 0.95, 
    dominant_num_batches: int = 10,
    label_format: str = "one-hot", # "index"
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
    target_transform = get_target_transform(target_type, scale_dict, label_format)

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

    # drop features
    if drop_dominant_features:
        keep_mask = compute_keep_mask_from_loader(
            train_loader,
            thr=dominant_thr,
            num_batches=dominant_num_batches,
        )

        print(f"[feature-drop] thr={dominant_thr}, batches={dominant_num_batches}")
        print(f"[feature-drop] dropped {(~keep_mask).sum().item()} / {keep_mask.numel()} features")

        train_dataset = MaskFeaturesDataset(train_dataset, keep_mask)
        val_dataset   = MaskFeaturesDataset(val_dataset, keep_mask)
        test_dataset  = MaskFeaturesDataset(test_dataset, keep_mask)

        train_loader = DataLoader(train_dataset, **loader_kwargs)
        val_loader   = DataLoader(val_dataset, **loader_kwargs)
        test_loader  = DataLoader(test_dataset, **loader_kwargs)

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
