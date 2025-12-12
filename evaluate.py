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
from torchvision.transforms import Compose, Lambda
from utils import SymLogTransform
import pandas as pd
from models.conv2d import Conv2DModel
import matplotlib.pyplot as plt
import seaborn as sns


def main(args):

    model = torch.load(args.model_path, weights_only=False)
    model.to(args.device)

    _, _, test_loader = create_dataloaders(
        args.data_config,
        batch_size=args.batch_size,
        shuffle=True,
        val_size=0.01,
        scaler_paths=('scaler_X.pkl', 'scaler_y.pkl'),
    )

    all_preds, all_labels = [], []
    model.eval()
    nmax = min(len(test_loader), 200)
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Evaluation", unit="batch", total=nmax)
        for step, (inputs, labels) in enumerate(pbar):
            if step > nmax:
                break
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            outputs = model(inputs)
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    print(f"all_preds shape: {all_preds.shape}, all_labels shape: {all_labels.shape}")

    label_names = test_loader.dataset.label_names
    for i in range(all_labels.shape[1]):
        var_name = label_names[i]
        plt.figure(figsize=(6, 4))
        plt.hist(all_labels[:, i], bins=50, alpha=0.5, label='True', color='blue')
        plt.hist(all_preds[:, i], bins=50, alpha=0.5, label='Predicted', color='orange')
        plt.title(f'Distribution of True vs Predicted for {var_name}')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.yscale('log')
        plt.show()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SmartPixels models")

    # Dataset and architecture
    parser.add_argument(
        "--data-config", type=str, default="config/v3.yml"
    )
    # parser.add_argument(
    #     "--architecture", "-a", type=str, default="conv", choices=["conv"], help="Model architecture."
    # )
    parser.add_argument(
        "--model-path", type=str, default="results/training/best_model.pth", help="Path to pre-trained model."
    )

    parser.add_argument("--batch-size", "-bs", type=int, default=128, help="Batch size")

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

    args = parser.parse_args()

    main(args)
