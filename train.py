import argparse
from pathlib import Path
import numpy as np
from glob import glob
from collections import defaultdict
from tensorboardX import SummaryWriter

import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils import CreateFolder, IsValidFile, IsReadableDir, EarlyStopping
from dataloader import create_dataloaders
from utils import get_device
import pandas as pd
from models.registry import REGISTRY
from datetime import datetime


def save_model(model, model_name, path):
    if "brevitas" in model_name:
        torch.save({
            "format": "registry_state_dict",
            "model": model_name,
            "state_dict": model.state_dict(),
        }, path)
    else:
        torch.save(model, path)


def get_roc_auc(preds, y):
    from sklearn.metrics import roc_auc_score

    # preds = preds.cpu().numpy()
    # y = y.cpu().numpy()

    # if y.shape[1] == 1:
    #     # Binary classification with single output neuron
    #     return roc_auc_score(y, preds)
    # else:
    y_class = y.argmax(dim=-1).cpu().numpy()
    probs = F.softmax(preds, dim=-1).cpu().numpy()
    # print(f"y_class.shape: {y_class.shape}, preds.shape: {preds.shape}")
    # score = roc_auc_score(y_class, probs, multi_class='ovr', average=None)
    # score = roc_auc_score(y_class, probs, multi_class='ovo', average=None)
    # print(f"ROC AUC score: {score}")
    return roc_auc_score(y_class, probs, multi_class='ovr')


def get_class_weights(data_loader):
    """Calculate class weights for imbalanced classification."""
    all_labels = data_loader.dataset.get_all_class_labels()
    class_counts = np.bincount(all_labels)
    total_samples = len(all_labels)
    num_classes = len(class_counts)

    # Calculate weights inversely proportional to class frequencies
    class_weights = total_samples / (num_classes * class_counts)
    return torch.tensor(class_weights, dtype=torch.float32)


def get_pos_class_weights(data_loader):
    """Calculate positive class weights for imbalanced multi-label classification."""
    all_labels = data_loader.dataset.get_all_class_labels()
    unique_labels = np.unique(all_labels)
    pos_weights = torch.ones(len(unique_labels), dtype=torch.float32)
    for i, label in enumerate(unique_labels):
        pos_weights[i] = (all_labels != label).sum() / (all_labels == label).sum()
    print(f"Positive class weights: {pos_weights}")
    return pos_weights


def derive_thresholds_from_data(data_loader, num_bits, one_per="feature", method="distributive"):
    """Derive thresholds for binarization from the training data.
    Args:
        data_loader: DataLoader providing the training data.
        num_bits: Number of bits for binarization.
        one_per: Whether to derive thresholds per feature or globally. Options: "feature", "global", "channel".
        method: Method to derive thresholds. Options: "distributive", "uniform".
    Returns:
        thresholds: Tensor of shape (num_features, num_bits) containing the derived thresholds.
    """
    from torchlogix.layers import Binarization

    data_set = []
    for i, (x, y) in enumerate(data_loader):
        if i >= 15:  # Use only a subset of the data for threshold derivation
            break
        data_set.append(x)
    data_set = torch.cat(data_set, dim=0)

    thresholds = Binarization.get_initial_thresholds(
        data_set,
        num_bits=num_bits,
        one_per=one_per,
        method=method
    )
    return thresholds


def evaluate_model(model, loader, eval_functions, mode="eval", device="cuda"):
    """Evaluate model on a data loader with given evaluation functions.
    Assumes metrics can be computed in batches and averaged."""
    orig_mode = model.training
    model.train(mode == "train")

    metrics = defaultdict(list)

    pbar = tqdm(loader, desc=f"Validation ({mode} mode)", leave=True, unit="batch")

    with torch.no_grad():
        for i, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)

            preds = model(x)

            for name, fn in eval_functions.items():
                vals = fn(preds, y)
                if isinstance(vals, torch.Tensor):
                    val = vals.mean().item()  # Average over the batch and convert to scalar
                else:
                    val = np.mean(vals)
                # metrics[name].append(fn(preds, y).to(torch.float32).mean().item())
                metrics[name].append(val)

            if i % 100 == 0:
                pbar.set_postfix(**{name: f"{np.mean(vals):.3f}" for name, vals in metrics.items()})

    model.train(orig_mode)

    return {name: np.mean(vals) for name, vals in metrics.items()}


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

    if args.filter:
        train_loader, val_loader, test_loader = create_dataloaders(
            args.data_config,
            batch_size=args.batch_size,
            shuffle=False,
            val_size=0.01,
            balance_classes=args.balance_classes,
            input_type=input_type,
            target_type=task_type,
            drop_dominant_features=True,
            dominant_thr=0.95, 
            dominant_num_batches=10, 
            )
    else:
        train_loader, val_loader, test_loader = create_dataloaders(
        args.data_config,
        batch_size=args.batch_size,
        shuffle=False,
        val_size=args.val_size,
        balance_classes=args.balance_classes,
        input_type=input_type,
        target_type=task_type,
        drop_dominant_features=False,
    )

    if args.plot_data and task_type != "classification":
        plot_data_distributions(train_loader)

    model_cls = model_cfg["class"]

    if "lgn" in args.model:
        thresholds = derive_thresholds_from_data(train_loader, num_bits=model_cls.n_bits)
        print(f"Derived thresholds for binarization: {thresholds}")
        model = model_cls(thresholds=thresholds)
    else:
        model = model_cls()

    print(model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {n_params}")

    # derive class weights for imbalanced classification
    if args.balance_classes and task_type == "classification":
        if args.loss_function == "cross_entropy":
            class_weights = get_class_weights(train_loader).to(device)
        elif args.loss_function == "bce":
            class_weights = get_pos_class_weights(train_loader).to(device)
    else:
        class_weights = None

    if task_type == "classification":
        if args.loss_function == "cross_entropy":
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        elif args.loss_function == "bce":
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights)
    elif task_type == "regression":
        criterion = torch.nn.MSELoss()
    else:
        raise ValueError(f"Unknown task: {task_type}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    model.to(device)

    eval_functions = {
        "loss": criterion,
        "acc": lambda preds, y: (preds.argmax(-1) == y.argmax(-1)).to(torch.float32).mean(),
        "balanced_acc": lambda preds, y: torch.stack([(preds.argmax(-1)[y.argmax(-1) == c] == c).to(torch.float32).mean() for c in range(y.shape[-1])]).mean(),
        # "roc_auc": get_roc_auc,
    }

    if args.compile_model:
        print("Compiling the model...")
        model.compile(dynamic=True)

    model.train()
    early_stopping = EarlyStopping(patience=200)
    best_val_loss = float('inf')

    dir_name = f"logs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    writer = SummaryWriter(logdir=f"{args.output}/{dir_name}", filename_suffix=".log")
    hparams = {
        "lr": args.learning_rate,
        "batch_size": args.batch_size,
        "model": args.model,
        "n_params": n_params,
    }

    step, running_loss, running_n, stop = 0, 0.0, 0, False
    epoch = 0
    while not stop and (args.epochs == 0 or epoch < args.epochs):
        train_loader.dataset.shuffle_indices()  # Shuffle the dataset at the beginning of each epoch
        pbar = tqdm(train_loader, desc=f"Training (Epoch {epoch})", unit="batch", leave=True)
        for inputs, labels in pbar:
            step += 1
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss
            running_n += 1
            if step % 100 == 0:
                preds = outputs.argmax(dim=1)  # Get the class with highest probability
                truth = labels.argmax(dim=1)  # Get the true class
                acc = (preds == truth).float().mean().item()
                pbar.set_postfix({
                    "running loss": f"{running_loss.item() / running_n:.3f}",
                    "acc": f"{acc:.3f}",
                    "loss": f"{loss.item():.3f}"
                })
                writer.add_scalar("Loss/train_batch", loss.item(), step)


            if args.eval_freq > 0 and step % args.eval_freq == 0:
                metrics = evaluate_model(
                    model, val_loader, eval_functions, mode="eval", device=args.device
                )
                for name, val in metrics.items():
                    writer.add_scalar(f"Discrete/{name}", val, step)
                if "lgn" in args.model:
                    relaxed_metrics = evaluate_model(
                        model, val_loader, eval_functions, mode="train", device=args.device
                    )
                    for name, val in relaxed_metrics.items():
                        writer.add_scalar(f"Relaxed/{name}", val, step)

                # if val_loss < best_val_loss:
                if metrics["loss"] < best_val_loss:
                    best_val_loss = metrics["loss"]
                    # writer.add_hparams(hparams, {"val_loss": val_loss})
                    # print(f"New best model found at step {step+1} with validation loss {val_loss:.4f}. Saving model...")
                    save_model(model, args.model, f"{args.output}/best_model.pth")
                # print(f"Step {step+1}/{len(train_loader)}, Validation Loss: {val_loss:.4f}")
                if not args.disable_early_stopping and early_stopping.step(metrics["loss"]):
                    stop = True
                    print("Early stopping triggered.")
                    break
                if args.steps is not None and step >= args.steps:
                    stop = True
                    print(f"Reached the maximum number of steps: {args.steps}. Stopping training.")
                    break
                model.train()
        epoch += 1

    save_model(model, args.model, f"{args.output}/final_model.pth")


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
        "--epochs", type=int, default=10, help="Number of epochs. Set to 0 to train until stopped."
    )
    parser.add_argument(
        "--balance-classes",  action='store_true', help="Balance the dataset classes during training (only for classification tasks)."
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.0, help="Weight decay for AdamW optimizer"
    )
    parser.add_argument(
        "--loss-function", "-lf", type=str, default="cross_entropy", choices=["cross_entropy", "bce"], help="Loss function to use for classification tasks"
    )

    parser.add_argument(
        "--filter", action = 'store_true', help="Choose to filter mostly non-varying fetures."
    )
    parser.add_argument(
        "--disable-early-stopping", "--no-early-stop", action="store_true",
        help="Disable validation-loss early stopping."
    )

    args = parser.parse_args()

    main(args)
