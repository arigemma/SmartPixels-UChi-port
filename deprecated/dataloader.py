"""Custom lazy-loading dataloader for SmartPixels signal/background classification."""

from .dataset import SmartPixelsDataset

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, List
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


def symlog_transform(x: np.ndarray, linthresh: float = 10.0) -> np.ndarray:
    """Symmetric log transform that handles negative values and zeros."""
    return np.sign(x) * np.log1p(np.abs(x) / linthresh)


def inverse_symlog_transform(x: np.ndarray, linthresh: float = 10.0) -> np.ndarray:
    """Inverse symmetric log transform."""
    return np.sign(x) * linthresh * (np.exp(np.abs(x)) - 1)


class LazySmartPixelsDataset(Dataset):
    """Lazy-loading PyTorch Dataset for SmartPixels 3D charge distribution data."""

    def __init__(
        self,
        file_indices: List[Tuple[str, str, int]],  # (label_file, image_file, row_idx)
        image_shape: Tuple[int, int, int] = (20, 13, 21),
        transform: bool = True,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        linthresh: float = 10.0
    ):
        """
        Args:
            file_indices: List of (label_file_path, image_file_path, row_idx) tuples
            image_shape: Shape to reshape images to (T, H, W)
            transform: Whether to apply symlog transformation and normalization
            mean: Mean for normalization (required if transform=True)
            std: Std for normalization (required if transform=True)
            linthresh: Threshold for symlog transform
        """
        self.file_indices = file_indices
        self.image_shape = image_shape
        self.transform = transform
        self.linthresh = linthresh
        self.mean = mean
        self.std = std

        if transform and (mean is None or std is None):
            raise ValueError("mean and std must be provided when transform=True")

    def __len__(self) -> int:
        return len(self.file_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        label_file, image_file, row_idx = self.file_indices[idx]

        # Read single row from parquet files
        label_df = pd.read_parquet(label_file)
        image_df = pd.read_parquet(image_file)

        # Get the specific row
        label = label_df.iloc[row_idx]['is_signal']
        image = image_df.iloc[row_idx].values.reshape(self.image_shape)

        # Apply transformations
        if self.transform:
            image = symlog_transform(image, linthresh=self.linthresh)
            image = (image - self.mean) / self.std

        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Add channel dim
        label = torch.tensor(float(label), dtype=torch.float32)

        return image, label


def create_balanced_indices(
    label_pattern: str,
    image_pattern: str,
    pt_threshold: float = 0.2,
    data_root: str = "."
) -> Tuple[List[Tuple[str, str, int]], pd.DataFrame]:
    """
    Create balanced indices for lazy loading without loading all data.

    Args:
        label_pattern: Glob pattern for label files
        image_pattern: Glob pattern for image files
        pt_threshold: Threshold for defining signal (|pt| > threshold)
        data_root: Root directory for data files

    Returns:
        file_indices: List of (label_file, image_file, row_idx) tuples
        all_labels: DataFrame with all labels (for computing statistics)
    """
    data_path = Path(data_root)
    label_files = sorted(data_path.glob(label_pattern))
    image_files = sorted(data_path.glob(image_pattern))

    if len(label_files) == 0 or len(image_files) == 0:
        raise ValueError(f"No files found matching patterns: {label_pattern}, {image_pattern}")

    if len(label_files) != len(image_files):
        raise ValueError(f"Mismatch: {len(label_files)} label files, {len(image_files)} image files")

    file_indices = []
    all_labels = []

    for label_file, image_file in zip(label_files, image_files):
        print(f"Processing: {label_file.name} & {image_file.name}")

        # Read labels only (much smaller than images)
        labels_df = pd.read_parquet(label_file)
        labels_df["is_signal"] = labels_df["pt"].abs() > pt_threshold
        labels_df["file_idx"] = len(all_labels)  # Track which file this came from

        # Balance dataset by undersampling majority class
        n_signal = labels_df["is_signal"].sum()
        n_background = (~labels_df["is_signal"]).sum()
        n_samples = min(n_signal, n_background)

        # Sample equal amounts and shuffle
        balanced_labels = pd.concat([
            labels_df[labels_df["is_signal"]].sample(n=n_samples, random_state=42),
            labels_df[~labels_df["is_signal"]].sample(n=n_samples, random_state=42)
        ]).sample(frac=1, random_state=42)

        # Create indices for this file
        for idx in balanced_labels.index:
            file_indices.append((str(label_file), str(image_file), idx))

        all_labels.append(balanced_labels.reset_index(drop=True))

    # Concatenate all labels
    all_labels_df = pd.concat(all_labels, axis=0, ignore_index=True)

    print(f"Created {len(file_indices)} balanced indices.")
    return file_indices, all_labels_df


def compute_normalization_stats(
    file_indices: List[Tuple[str, str, int]],
    image_shape: Tuple[int, int, int] = (20, 13, 21),
    linthresh: float = 10.0,
    sample_size: int = 1000
) -> Tuple[float, float]:
    """
    Compute normalization statistics by sampling a subset of data.

    Args:
        file_indices: List of file indices
        image_shape: Shape to reshape images to
        linthresh: Threshold for symlog transform
        sample_size: Number of samples to use for computing statistics

    Returns:
        mean: Mean of transformed data
        std: Standard deviation of transformed data
    """
    print(f"Computing normalization stats from {sample_size} samples...")

    # Sample random indices
    np.random.seed(42)
    sample_indices = np.random.choice(len(file_indices), size=min(sample_size, len(file_indices)), replace=False)

    samples = []
    for idx in sample_indices:
        label_file, image_file, row_idx = file_indices[idx]
        image_df = pd.read_parquet(image_file)
        image = image_df.iloc[row_idx].values.reshape(image_shape)
        image = symlog_transform(image, linthresh=linthresh)
        samples.append(image.flatten())

    samples = np.concatenate(samples)
    mean = samples.mean()
    std = samples.std()

    print(f"Normalization stats: mean={mean:.4f}, std={std:.4f}")
    return float(mean), float(std)


def create_dataloaders(
    label_pattern: str,
    image_pattern: str,
    batch_size: int = 32,
    test_size: float = 0.2,
    pt_threshold: float = 0.2,
    linthresh: float = 10.0,
    data_root: str = ".",
    num_workers: int = 0,
    sample_size_for_stats: int = 1000
) -> Tuple[DataLoader, DataLoader, Tuple[float, float]]:
    """
    Create lazy-loading train and test dataloaders.

    Args:
        label_pattern: Glob pattern for label files
        image_pattern: Glob pattern for image files
        batch_size: Batch size for dataloaders
        test_size: Fraction of data for test set
        pt_threshold: Threshold for defining signal
        linthresh: Threshold for symlog transform
        data_root: Root directory for data files
        num_workers: Number of workers for dataloaders
        sample_size_for_stats: Number of samples for computing normalization stats

    Returns:
        train_loader: Training dataloader
        test_loader: Test dataloader
        normalization_stats: (mean, std) tuple
    """
    # Create balanced indices
    file_indices, labels_df = create_balanced_indices(
        label_pattern, image_pattern, pt_threshold, data_root
    )

    # Create binary labels for stratification
    binary_labels = labels_df['is_signal'].astype(int).values

    # Train-test split of indices
    train_indices, test_indices, y_train, y_test = train_test_split(
        file_indices, binary_labels,
        test_size=test_size,
        random_state=42,
        stratify=binary_labels
    )

    print(f"Train samples: {len(train_indices)}, Test samples: {len(test_indices)}")
    print(f"y_train distribution:\n{pd.Series(y_train).value_counts()}")
    print(f"y_test distribution:\n{pd.Series(y_test).value_counts()}")

    # Compute normalization statistics from training set
    mean, std = compute_normalization_stats(
        train_indices,
        linthresh=linthresh,
        sample_size=sample_size_for_stats
    )

    # Create datasets
    train_dataset = LazySmartPixelsDataset(
        train_indices,
        transform=True,
        mean=mean,
        std=std,
        linthresh=linthresh
    )

    test_dataset = LazySmartPixelsDataset(
        test_indices,
        transform=True,
        mean=mean,
        std=std,
        linthresh=linthresh
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, test_loader, (mean, std)
