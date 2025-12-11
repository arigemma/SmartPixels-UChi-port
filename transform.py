# transform.py
import numpy as np
from tqdm import tqdm


# -------------------------------------------------------
#  WELFORD ONLINE SCALER
# -------------------------------------------------------

def update_running_stats(x, count, mean, M2):
    """
    One Welford update step.
    x must be a 1D numpy array.
    """
    count += 1
    delta = x - mean
    mean += delta / count
    M2 += delta * (x - mean)
    return count, mean, M2


def compute_dataset_scaler(dataset, max_samples=None):
    """
    Iterates once over the dataset and computes mean/std for inputs and targets.

    max_samples: optionally only process the first N samples.
    """

    # Initialize with first sample
    first_x, first_y = dataset[0]
    first_x = np.asarray(first_x).ravel()
    first_y = np.asarray(first_y).ravel()

    # x stats
    x_count = 0
    x_mean = np.zeros_like(first_x, dtype=np.float64)
    x_M2   = np.zeros_like(first_x, dtype=np.float64)

    # y stats
    y_count = 0
    y_mean = np.zeros_like(first_y, dtype=np.float64)
    y_M2   = np.zeros_like(first_y, dtype=np.float64)

    iterator = tqdm(dataset, desc="Computing scaling parameters")

    for i, (x, y) in enumerate(iterator):
        if max_samples is not None and i >= max_samples:
            break

        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()

        x_count, x_mean, x_M2 = update_running_stats(x, x_count, x_mean, x_M2)
        y_count, y_mean, y_M2 = update_running_stats(y, y_count, y_mean, y_M2)

    # Finalize
    x_var = x_M2 / (x_count - 1)
    y_var = y_M2 / (y_count - 1)

    x_std = np.sqrt(x_var)
    y_std = np.sqrt(y_var)

    return (x_mean, x_std), (y_mean, y_std)


def save_scaler(x_stats, y_stats, prefix="scaler"):
    x_mean, x_std = x_stats
    y_mean, y_std = y_stats

    np.save(f"{prefix}_x_mean.npy", x_mean)
    np.save(f"{prefix}_x_std.npy",  x_std)
    np.save(f"{prefix}_y_mean.npy", y_mean)
    np.save(f"{prefix}_y_std.npy",  y_std)

    print(f"Scaler parameters saved with prefix '{prefix}'")


def load_scaler(prefix="scaler"):
    x_mean = np.load(f"{prefix}_x_mean.npy")
    x_std  = np.load(f"{prefix}_x_std.npy")
    y_mean = np.load(f"{prefix}_y_mean.npy")
    y_std  = np.load(f"{prefix}_y_std.npy")
    return (x_mean, x_std), (y_mean, y_std)


# -------------------------------------------------------
#  TRANSFORMS
# -------------------------------------------------------

class Normalize:
    """Normalize input features: x -> (x - mean) / std"""
    def __init__(self, mean, std):
        self.mean = mean
        self.std  = std

    def __call__(self, x):
        x = np.asarray(x).ravel()
        return (x - self.mean) / self.std


class NormalizeTarget:
    """Normalize target features: y -> (y - mean) / std"""
    def __init__(self, mean, std):
        self.mean = mean
        self.std  = std

    def __call__(self, y):
        y = np.asarray(y).ravel()
        return (y - self.mean) / self.std


# -------------------------------------------------------
#  MAIN: COMPUTE AND SAVE SCALING PARAMETERS
# -------------------------------------------------------

if __name__ == "__main__":
    import yaml
    import argparse
    from dataset import SmartPixelsDataset

    parser = argparse.ArgumentParser(description="Compute normalization parameters for SmartPixels dataset.")
    parser.add_argument(
        "--data-config", type=str, default="config/v3.yml"
    )
    parser.add_argument("--prefix", type=str, default="scaler", help="Output file prefix.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional max number of samples to use.")
    args = parser.parse_args()

    config_dict = yaml.safe_load(open(args.data_config, "r"))

    image_paths = [config_dict['root'] + t[0] for t in config_dict['train'] + config_dict['test']]
    label_paths = [config_dict['root'] + t[1] for t in config_dict['train'] + config_dict['test']]

    ds = SmartPixelsDataset(
        image_paths,
        label_paths,
        shuffle=False
    )

    # Compute normalization parameters
    x_stats, y_stats = compute_dataset_scaler(ds, max_samples=args.max_samples)

    # Save them
    save_scaler(x_stats, y_stats, prefix=args.prefix)
    print("Done.")
