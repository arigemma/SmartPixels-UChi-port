import argparse
import random
from pathlib import Path
from collections import defaultdict
import numpy as np
import hist
from glob import glob
from tensorboardX import SummaryWriter

import torch
from tqdm import tqdm

from utils import CreateFolder, IsValidFile, IsReadableDir, EarlyStopping
from dataloader import create_dataloaders
import joblib
from utils import SymLogTransform
import pandas as pd
from models import get_model
from datetime import datetime


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

    if args.device == "mps":
        device = torch.device("mps")

    train_loader, val_loader, test_loader = create_dataloaders(
        args.data_config,
        batch_size=args.batch_size,
        shuffle=True,
        val_size=0.01,
        scaler_paths=('scaler_X.pkl', 'scaler_y.pkl'),
    )

    if args.plot_data:
        plot_data_distributions(train_loader)

    model = get_model(args.architecture, size=args.model_size)
    model.to(device)
    print(model)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {n_params}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.MSELoss()

    model.train()
    running_loss = 0.0
    early_stopping = EarlyStopping(patience=2)
    best_val_loss = float('inf')

    dir_name = f"logs/{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    writer = SummaryWriter(logdir=f"{args.output}/{dir_name}", filename_suffix=".log")
    hparams = {
        "lr": args.learning_rate,
        "batch_size": args.batch_size,
        "model": args.architecture,
        "model_size": args.model_size,
        "n_params": n_params,
    }

    pbar = tqdm(train_loader, desc="Training", unit="batch")
    for step, (inputs, labels) in enumerate(pbar):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        pbar.set_postfix({"loss": loss.item()})
        writer.add_scalar("Loss/train_batch", loss.item(), step)

        if step % args.eval_freq == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in tqdm(val_loader):
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
            val_loss = val_loss / len(val_loader.dataset)
            writer.add_scalar("Loss/validation", val_loss, step)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                writer.add_hparams(hparams, {"val_loss": val_loss})
                torch.save(model, args.output / "best_model.pth")
            print(f"Step {step+1}/{len(train_loader)}, Validation Loss: {val_loss:.4f}")
            if early_stopping.step(val_loss):
                print("Early stopping triggered.")
                break
            model.train()

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SmartPixels models")

    # Dataset and architecture
    parser.add_argument(
        "--data-config", type=str, default="config/v3.yml"
    )
    parser.add_argument(
        "--architecture", "-a", type=str, default="conv", choices=["conv2d", "conv3d"], help="Model architecture."
    )
    parser.add_argument("--model-size", "-ms", type=int, default=4, 
                        help="Model size parameter (increases number of parameters)")

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
        "--device", type=str, default="cpu", choices=["cuda", "cpu", "mps"],
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
