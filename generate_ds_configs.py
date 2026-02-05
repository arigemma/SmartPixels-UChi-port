from glob import glob
import numpy as np
import yaml
import os
from pathlib import Path
from dataset import SmartPixelsDataset
from utils import SymLogTransform
from argparse import ArgumentParser


def get_x_and_y(image_path, label_path):
    ds = SmartPixelsDataset(
        [image_path],
        [label_path],
        shuffle=False,
        # important: pre-transform with symlog
        transform=SymLogTransform(),
    )

    X, Y = [], []
    for i, (x, y) in enumerate(ds):
        X.append(x)
        Y.append(y)
    return np.stack(X), np.stack(Y)


def main(args):
    ds_to_path = {
        "barrel-phys-pt": "dataset_2s",
        "barrel-flat-pt": "dataset_3s",
        "endcap-phys-pt": "dataset_6s",
        "endcap-flat-pt": "dataset_7s",
        "irradiated-barrel-phys-pt": "dataset_8s",
        "irradiated-barrel-flat-pt": "dataset_9s",
    }

    base_path = args.data_root

    for ds_name, ds_folder in ds_to_path.items():
        ds_path = base_path + f"/{ds_folder}/"
        for sub_folder in glob(ds_path + "*/"):
            sub_name = sub_folder.split("/")[-2].split("_")[-1]
            label_files = sorted(glob(sub_folder + "label*.parquet"))
            reco_files = sorted(glob(sub_folder + "reco*.parquet"))
            n_test = len(label_files) // 3
            label_train_files = label_files[:-n_test]
            label_test_files = label_files[-n_test:]
            reco_train_files = reco_files[:-n_test]
            reco_test_files = reco_files[-n_test:]

            # Get mean and std for each feature (X after symlog)
            X, y = get_x_and_y(reco_train_files[0], label_train_files[0])
            X = X.reshape(-1, 1)
            mean_X = np.mean(X, axis=0)
            std_X = np.std(X, axis=0)

            print(f"Dataset: {ds_name}, Subset: {sub_name}")
            print(f"  mean_X: {mean_X.tolist()}")
            print(f"  std_X: {std_X.tolist()}")

            mean_y = np.mean(y, axis=0)
            std_y = np.std(y, axis=0)

            # save file_dict as yaml
            # create dict <ds_name> if doesnt exist
            if not os.path.exists(f"{args.output_dir}/{ds_name}"):
                os.makedirs(f"{args.output_dir}/{ds_name}")

            with open(f"{args.output_dir}/{ds_name}/{sub_name}.yml", "w") as f:
                f.write(f"root: {sub_folder}\n")
                f.write("train:\n")
                for r, l in zip(reco_train_files, label_train_files):
                    f.write(f"- [{Path(r).name}, {Path(l).name}]\n")
                f.write("test:\n")
                for r, l in zip(reco_test_files, label_test_files):
                    f.write(f"- [{Path(r).name}, {Path(l).name}]\n")

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
        "--data-root", "-d", type=str, default="/scratch/gpfs/IOJALVO/lg0508/SmartPixels/17180303",
        help="Root directory containing SmartPixels datasets."
    )
    parser.add_argument(
        "--output-dir", "-o", type=str, default="dataset_configs",
        help="Output directory to save dataset configuration files."
    )
    args = parser.parse_args()
    main(args)
    