import argparse
import random
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from utils import CreateFolder, IsValidFile, IsReadableDir, EarlyStopping
from dataloader import create_dataloaders
import joblib
from torchvision.transforms import Compose, Lambda
import pandas as pd
import seaborn as sns
import math
import matplotlib.pyplot as plt
from models.registry import REGISTRY, class_to_config, load_registered_model


def main(args):

    model = load_registered_model(args.model_path)
    model.to(args.device)

    name, model_cfg = class_to_config(type(model))

    print(f"Loaded model from {args.model_path}, which corresponds to registry entry '{name}' with input type <{model_cfg['input_type']}> and task type <{model_cfg['task_type']}>")

    input_type = model_cfg["input_type"]
    task_type = model_cfg["task_type"]

    _, _, test_loader = create_dataloaders(
        args.data_config,
        batch_size=args.batch_size,
        shuffle=True,
        val_size=0.01,
        input_type=input_type,
        target_type=task_type,
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

    if task_type == "classification":
        # For classification, we can do confusion matrices or ROC curves
        from sklearn.metrics import confusion_matrix, roc_curve, auc

        n_classes = all_labels.shape[1]
        for i in range(n_classes):
            cm = confusion_matrix(all_labels[:, i], all_preds[:, i] > 0.5)
            print(f"Confusion matrix for class {i}:\n{cm}")
            # plot confusion matrix
            plt.figure()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.savefig(f'{args.output}/confusion_matrix_class_{i}.png')

            fpr, tpr, _ = roc_curve(all_labels[:, i], all_preds[:, i])
            roc_auc = auc(fpr, tpr)
            plt.figure()
            plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve for Class {i}')
            plt.legend(loc='lower right')
            plt.savefig(f'{args.output}/roc_curve_class_{i}.png')


    else:
        label_names = test_loader.dataset.label_names

        label_names = test_loader.dataset.label_names
        n = len(label_names)

        vars_per_fig = 8        # threshold
        cols = 4                # hist | scatter pairs
        plots_per_var = 2

        # number of figures needed
        num_figs = math.ceil(n / vars_per_fig)

        start_idx = 0
        for fig_idx in range(num_figs):
            # variables for this figure
            end_idx = min(start_idx + vars_per_fig, n)
            num_vars = end_idx - start_idx

            rows = math.ceil(num_vars * plots_per_var / cols)

            fig, axs = plt.subplots(rows, cols, figsize=(16, rows * 3))
            axs = axs.reshape(rows, cols)

            fig.suptitle(f"Variable Diagnostics (Figure {fig_idx+1}/{num_figs})",
                        fontsize=16, y=1.02)

            for local_i, global_i in enumerate(range(start_idx, end_idx)):
                r = (local_i // 2)
                c = (local_i % 2) * 2

                ax1 = axs[r, c]
                ax2 = axs[r, c+1]

                var_name = label_names[global_i]

                h = ax1.hist(all_labels[:, global_i], bins=50, alpha=0.5, label='True')
                ax1.hist(all_preds[:, global_i], bins=h[1], alpha=0.5, label='Predicted')
                ax1.set_title(f'{var_name} – Distribution')
                ax1.set_yscale('log')
                ax1.legend()

                ax2.scatter(all_labels[:, global_i], all_preds[:, global_i], alpha=0.5)
                ax2.set_title(f'{var_name} – Scatter')
                ax2.set_xscale('symlog')
                ax2.set_yscale('symlog')

            # hide any unused subplots
            for ax in axs.flatten()[num_vars*2:]:
                ax.set_visible(False)

            plt.tight_layout()
            plt.show()

            start_idx = end_idx

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SmartPixels models")

    # Dataset and architecture
    parser.add_argument(
        "--data-config", "-c", type=Path, default="config/10783560/v3.yml"
    )
    parser.add_argument(
        "--model-path", type=str, default="results/training/best_model.pth", help="Path to pre-trained model."
    )

    parser.add_argument("--batch-size", "-bs", type=int, default=128, help="Batch size")

    parser.add_argument(
        "--device", type=str, default="cpu", choices=["cuda", "cpu", "mps"],
        help="Device to use (cuda is faster)"
    )

    parser.add_argument(
        "--output", "-o", action=CreateFolder, type=Path, default="results/evaluation/",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--verbose", action='store_true', help="Enable verbose output"
    )

    args = parser.parse_args()

    main(args)
