import argparse
from pathlib import Path
import numpy as np
from glob import glob
from tensorboardX import SummaryWriter

import torch
from tqdm import tqdm

from utils import CreateFolder, IsValidFile, IsReadableDir, EarlyStopping
from dataloader import create_dataloaders
from utils import get_device
import pandas as pd
from models.registry import REGISTRY
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
    plt.savefig("data_distribution_X.png")
    plt.clf()

    if y.shape[1] == len(data_loader.dataset.label_names):
        label_names = data_loader.dataset.label_names
    else:
        label_names = [str(i) for i in range(y.shape[1])]

    df = pd.DataFrame(Y, columns=label_names)

    # Use seaborn's default style
    sns.set_style("whitegrid")

    # Create the histograms
    df.hist(figsize=(12, 10), bins=20, edgecolor='black')
    plt.tight_layout()
    plt.savefig("data_distribution_Y.png")
    plt.clf()


def main(args):

    # set seed for reproducibility
    torch.manual_seed(args.seed)

    device = get_device(args.device)

    model_cfg = REGISTRY[args.model]
    print(f"Selected model: {args.model}, which expects input type <{model_cfg['input_type']}> and is designed for task <{model_cfg['task_type']}>")

    input_type = model_cfg["input_type"]
    task_type = model_cfg["task_type"]

    if task_type == "regression" and args.balance_classes:
        raise ValueError("Balancing classes is not supported for regression tasks.")

    train_loader, val_loader, test_loader = create_dataloaders(
        args.data_config,
        batch_size=args.batch_size,
        shuffle=False,
        val_size=args.val_size,
        balance_classes=args.balance_classes,
        input_type=input_type,
        target_type=task_type,
        label_format="index",
    )

    if args.plot_data and task_type != "classification":
        plot_data_distributions(train_loader)

    model = model_cfg["class"]()

    print(model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {n_params}")

    if task_type == "classification":
        # criterion = torch.nn.BCEWithLogitsLoss()
        criterion = torch.nn.CrossEntropyLoss()
    elif task_type == "regression":
        criterion = torch.nn.MSELoss()
    else:
        raise ValueError(f"Unknown task: {task_type}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    model.to(device)

    if args.compile_model:
        print("Compiling the model...")
        model.compile()

    model.train()
    early_stopping = EarlyStopping(patience=20)
    best_val_loss = float('inf')

    dir_name = f"logs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    writer = SummaryWriter(logdir=f"{args.output}/{dir_name}", filename_suffix=".log")
    hparams = {
        "lr": args.learning_rate,
        "batch_size": args.batch_size,
        "model": args.model,
        "n_params": n_params,
    }


    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        running_loss, running_n = 0.0, 0

        pbar = tqdm(train_loader, desc=f"Training epoch {epoch+1}/{args.epochs}", unit="batch")
        for step_in_epoch, (inputs, labels) in enumerate(pbar):
            step = global_step 
            global_step += 1

            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss
            running_n += 1

            if step % 10 == 0:
                pbar.set_postfix({
                    "running loss": f"{running_loss.item() / running_n:.3f}",
                    "loss": f"{loss.item():.3f}"
                })
                writer.add_scalar("Loss/train_batch", loss.item(), step)

            if args.eval_freq > 0 and step % args.eval_freq == 0:
                model.eval()
                val_loss, val_acc, val_n = 0.0, 0.0, 0
                with torch.no_grad():
                    pbar2 = tqdm(val_loader, desc="Validation", leave=True, unit="batch")
                    for inputs, labels in pbar2:
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        val_loss += loss
                        if task_type == "classification":
                            preds = outputs.argmax(dim=1)
                            val_acc += (preds == labels).float().mean().item()
                        val_n += 1
                        pbar2.set_postfix({
                            "running val loss": f"{val_loss.item() / val_n:.3f}",
                            "running val acc": f"{val_acc / val_n:.3f}",
                            "val loss": f"{loss.item():.3f}",
                        })

                writer.add_scalar("Loss/validation", val_loss.item() / val_n, step)
                writer.add_scalar("Accuracy/validation", val_acc / val_n, step)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model, f"{args.output}/best_model.pth")

                if early_stopping.step(val_loss):
                    print("Early stopping triggered.")
                    return  

                model.train()

            # keep step limit feature working from before
            if args.steps is not None and global_step >= args.steps:
                print(f"Reached the maximum number of steps: {args.steps}. Stopping training.")
                return  


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SmartPixels models")

    # Dataset and architecture
    parser.add_argument(
        "--data-config", "-c", type=str, default="config/10783560/v3.yml"
    )
    parser.add_argument(
        "--model", "-m", type=str, default="towards-model-2", choices=list(REGISTRY.keys()), help="Model architecture."
    )

    # Training parameters
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", "-bs", type=int, default=1024, help="Batch size")
    parser.add_argument("--learning-rate", "-lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--val-size", type=float, default=0.01)
    parser.add_argument(
        "--num-iterations", "-ni", type=int, default=100_000, help="Number of training iterations"
    )
    parser.add_argument(
        "--eval-freq", "-ef", type=int, default=200, 
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
    parser.add_argument(
        "--compile-model", action='store_true', help="Compile the model with torch.compile"
    )
    parser.add_argument(
        "--steps", type=int, default=None, help="Number of training steps. If not set, uses full dataset."
    )
    parser.add_argument(
        "--balance-classes",  action='store_true', help="Balance the dataset classes during training (only for classification tasks)."
    )
    
    args = parser.parse_args()

    main(args)
