import argparse
import random
from pathlib import Path
from collections import defaultdict
import numpy as np
import hist
from glob import glob

import torch
from tqdm import tqdm

from utils import CreateFolder, IsValidFile, IsReadableDir
from dataloader import create_dataloaders


def symlog_transform(x: np.ndarray, linthresh: float = 10.0) -> np.ndarray:
    """Symmetric log transform that handles negative values and zeros."""
    return np.sign(x) * np.log1p(np.abs(x) / linthresh)


def inverse_symlog_transform(x: np.ndarray, linthresh: float = 10.0) -> np.ndarray:
    """Inverse symmetric log transform."""
    return np.sign(x) * linthresh * (np.exp(np.abs(x)) - 1)


def main(args):
    image_paths = sorted(glob("data/10783560/recon3D/recon3D_d*.parquet"))
    label_paths = sorted(glob("data/10783560/labels/labels_d**.parquet"))

    train_loader, val_loader, test_loader = create_dataloaders(
        args.data_config,
        batch_size=args.batch_size,
        shuffle=True,
        transform=None,
        target_transform=None,
    )

    print(train_loader)
    print(val_loader)
    print(test_loader)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SmartPixels models")

    # Dataset and architecture
    parser.add_argument(
        "--data-config", type=str, default="config/v3.yml"
    )
    parser.add_argument(
        "--architecture", "-a", type=str, default="conv", choices=["conv"], help="Model architecture."
    )

    # Training parameters
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed")
    parser.add_argument("--batch-size", "-bs", type=int, default=128, help="Batch size")
    parser.add_argument("--learning-rate", "-lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--num-iterations", "-ni", type=int, default=100_000, help="Number of training iterations"
    )
    parser.add_argument(
        "--eval-freq", "-ef", type=int, default=2_000, 
        help="Evaluation frequency. Evaluation is deactivated if set to 0."
    )

    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cuda", "cpu", "mps"],
        help="Device to use (cuda is faster)"
    )

    parser.add_argument(
        "--output", "-o", action=CreateFolder, type=Path, default="results/training/",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--verbose", type=int, default=0, choices=[0, 1]
    )

    args = parser.parse_args()

    main(args)
