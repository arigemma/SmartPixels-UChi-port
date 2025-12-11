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
import joblib
from torchvision.transforms import Compose, Lambda
from utils import SymLogTransform
import pandas as pd



def plot_data_distributions(data_loader):
    import matplotlib.pyplot as plt
    import seaborn as sns

    n_max = 100
    X, Y = [], []
    for i, (x, y) in enumerate(tqdm(data_loader)):
        if i >= n_max:
            break
        X.append(x)
        Y.append(y)

    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)

    plt.hist(X.flatten(), bins=100, alpha=0.5, label='X')
    plt.yscale('log')
    plt.show()

    df = pd.DataFrame(Y)
    # Use seaborn's default style
    sns.set_style("whitegrid")

    # Create the histograms
    df.hist(figsize=(12, 10), bins=20, edgecolor='black')
    plt.tight_layout()
    plt.show()


def main(args):

    # 3. Load both scalers
    scaler_X = joblib.load('scaler_X.pkl')
    scaler_y = joblib.load('scaler_y.pkl')

    transform = Compose([
        SymLogTransform(),
        Lambda(lambda x: scaler_X.transform(x.reshape(-1, 1))),
        Lambda(lambda x: x.reshape(20, 13, 21)),
    ])

    target_transform = Lambda(lambda y: scaler_y.transform(y.reshape(1, -1)).flatten())

    train_loader, val_loader, test_loader = create_dataloaders(
        args.data_config,
        batch_size=args.batch_size,
        shuffle=True,
        transform=transform,
        target_transform=target_transform,
    )

    if args.plot_data:
        plot_data_distributions(train_loader)



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

    parser.add_argument(
        "--plot-data", action='store_true', help="Plot data distributions after loading"
    )

    args = parser.parse_args()

    main(args)
