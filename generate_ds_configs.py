from glob import glob
import numpy as np
import yaml
import os
from pathlib import Path
from dataset import SmartPixelsDataset
from utils import SymLogTransform
from argparse import ArgumentParser


def get_mean_and_std_for_x_and_y(image_path, label_path):
    ds = SmartPixelsDataset(
        [image_path],
        [label_path],
        shuffle=False,
        # important: pre-transform with symlog
        transform=lambda x, y: SymLogTransform()(x),
    )

    X, Y = [], []
    for i, (x, y) in enumerate(ds):
        X.append(x)
        Y.append(y)
    
    X, Y = np.stack(X), np.stack(Y)

    X = X.reshape(-1, 1)

    mean_X = np.mean(X, axis=0)
    std_X = np.std(X, axis=0)

    mean_y = np.mean(Y, axis=0)
    std_y = np.std(Y, axis=0)

    return mean_X, std_X, mean_y, std_y


def get_file_dict_17180303(base_path):
    """
    split dataset as defined on zenodo (https://zenodo.org/records/17180303)
    into human-readable names and list all files
    returns dict of shape:
    {
        dataset_name: (base_path, [reco_files], [label_files])
    }
    """
    base_path = Path(base_path) / "17180303"
    ds_to_path = {
        "barrel-phys-pt": "dataset_2s",
        "barrel-flat-pt": "dataset_3s",
        "endcap-phys-pt": "dataset_6s",
        "endcap-flat-pt": "dataset_7s",
        "irradiated-barrel-phys-pt": "dataset_8s",
        "irradiated-barrel-flat-pt": "dataset_9s",
    }
    file_list_dict = {}
    for ds_name, ds_folder in ds_to_path.items():
        ds_path = base_path / ds_folder
        for sub_folder in sorted(ds_path.iterdir()):
            sub_name = sub_folder.name.split("_")[-1]
            reco_files = sorted(str(f) for f in sub_folder.glob("reco*.parquet"))
            label_files = sorted(str(f) for f in sub_folder.glob("label*.parquet"))
            reco_files = [Path(f).relative_to(sub_folder) for f in reco_files]
            label_files = [Path(f).relative_to(sub_folder) for f in label_files]
            file_list_dict[f"17180303/{ds_name}/{sub_name}"] = (
                str(sub_folder),
                reco_files,
                label_files,
            )
    return file_list_dict


def get_file_dict_10783560(base_path):
    """
    split dataset as defined on zenodo (https://zenodo.org/records/17180303)
    into human-readable names and list all files
    returns dict of shape:
    {
        dataset_name: (base_path, [reco_files], [label_files])
    }
    """
    base_path = Path(base_path) / "10783560"
    reco_files = sorted(base_path.glob("recon3D/recon3D_*.parquet"))
    label_files = sorted(base_path.glob("labels/labels_*.parquet"))
    reco_files = [p.relative_to(base_path) for p in reco_files]
    label_files = [p.relative_to(base_path) for p in label_files]
    return {"10783560/v3": (base_path, reco_files, label_files)}


def main(args):

    file_dict = get_file_dict_10783560(args.data_root)
    # file_dict.update(get_file_dict_17180303(args.data_root))

    for ds_name, (root, reco_files, label_files) in file_dict.items():
        # get number of train- and test files
        n_test = int(len(label_files) * args.test_ratio)

        # split files into groups
        label_train_files = label_files[:-n_test]
        label_test_files = label_files[-n_test:]
        reco_train_files = reco_files[:-n_test]
        reco_test_files = reco_files[-n_test:]

        # Get mean and std for each feature (X after symlog)
        mean_X, std_X, mean_y, std_y = get_mean_and_std_for_x_and_y(
            f"{root}/{reco_train_files[0]}", f"{root}/{label_train_files[0]}"
        )

        config_dir = Path(f"{args.output_dir}/{ds_name}").parent
        if not config_dir.exists():
            os.makedirs(config_dir)

        with open(f"{args.output_dir}/{ds_name}.yml", "w") as f:
            f.write(f"root: {root}/\n")
            f.write("train:\n")
            for r, l in zip(reco_train_files, label_train_files):
                f.write(f"- [{r}, {l}]\n")
            f.write("test:\n")
            for r, l in zip(reco_test_files, label_test_files):
                f.write(f"- [{r}, {l}]\n")

            f.write("scaling:\n")
            f.write(f"  mean_X: {mean_X.tolist()}\n")
            f.write(f"  std_X: {std_X.tolist()}\n")
            f.write(f"  mean_y: {mean_y.tolist()}\n")
            f.write(f"  std_y: {std_y.tolist()}\n")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="""
        Derive dataset configuration files with scaling parameters for different SmartPixels datasets.
        1. For each dataset, count the number of samples and split it into train and test sets (custom ratio).
        2. Use the first file in the training set to compute mean and std for input features (after symlog transform) and target variables.
        3. Save configuration files in YAML format containing file paths and scaling parameters.
        """
    )
    parser.add_argument(
        "--data-root", "-d", type=str, default="/scratch/gpfs/IOJALVO/lg0508/SmartPixels/",
        help="Root directory containing SmartPixels datasets."
    )
    parser.add_argument(
        "--output-dir", "-o", type=str, default="config",
        help="Output directory to save dataset configuration files."
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.2,
        help="Ratio of the dataset to use for testing."
    )
    args = parser.parse_args()
    main(args)
