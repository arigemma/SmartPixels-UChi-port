import random
from typing import List
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import pandas as pd


class UnderSampler(Sampler):
    """
    Custom sampler to perform undersampling of classes in a dataset based on specified target class ratios.
     - dataset: The dataset to sample from (must implement get_all_class_labels())
     - target_class_ratios: mapping class labels to their desired ratios after sampling (e.g. {0: 0.5, 1: 0.3, 2: 0.2}).
    Determines the sampling probabilities to achieve the target ratios, and samples indices accordingly.
    Sampling probabilities are scaled such that the rarest class is always fully included (probability 1).
    Resulting indices are sorted to maintain caching efficiency.
    """

    def __init__(self, dataset: Dataset, target_class_ratios: dict):
        self.dataset = dataset
        self.target_class_ratios = target_class_ratios
        self.indeces = []

        all_labels = self.dataset.get_all_class_labels()

        unique_labels, counts = np.unique(all_labels, return_counts=True)

        assert set(target_class_ratios.keys()) == set(all_labels), \
            "Class probabilities must be provided for all classes in the dataset." \
            "Dataset classes: {}, Provided classes: {}".format(set(all_labels), set(target_class_ratios.keys()))

        existing_ratios = {label: count / len(all_labels) for label, count in zip(unique_labels, counts)}

        # calculate the sampling probabilities for each class
        class_probs = {label: target_class_ratios[label] / existing_ratios[label] for label in unique_labels}

        # identify the largest one and scale all such that the largest becomes 1
        max_prob = max(class_probs.values())
        class_probs = {label: prob / max_prob for label, prob in class_probs.items()}

        for label, prob in class_probs.items():
            label_idx = (all_labels == label)
            n_sampled = int(prob * np.sum(label_idx))
            sampled = np.random.choice(np.where(label_idx)[0], size=n_sampled, replace=False)
            self.indeces.append(sampled)

        # sort the indeces to maintain caching efficiency (recall: the dataset is already shuffled)
        self.indeces = np.concatenate(self.indeces)
        self.indeces.sort()

    def __len__(self):
        return len(self.indeces)

    def __iter__(self):
        return iter(self.indeces)


class SmartPixelsDataset(Dataset):
    """
    PyTorch Dataset for SmartPixels 3D charge distribution data with lazy loading and single-file caching.
    Using <image> to refer to the 3D charge distribution data and <label> to refer to the associated labels.
    Shuffling shuffles the files and the indices within each file such that caching is still effective.
    Make sure to provide matching lists of image and label file paths, e.g.
         [image1.parquet, image2.parquet], [label1.parquet, label2.parquet]
    Args:
        image_paths: List of image file paths
        label_paths: List of label file paths

        transform: Optional transform to be applied on a sample.
        target_transform: Optional transform to be applied on the target.
    """
    def __init__(
            self, image_paths: List[str], label_paths: List[str], shuffle: bool=False, transform=None, target_transform=None
        ):
        assert len(image_paths) == len(label_paths), "Number of image and label files must match."

        self.image_paths = image_paths
        self.label_paths = label_paths
        self.shuffle = shuffle
        self.transform = transform
        self.target_transform = target_transform

        self.label_names = None  # to be populated on first access

        # Build a global index_dict: {file_index: [internal_indices]}
        self.index_dict = {}
        for i, f in enumerate(self.label_paths):  # use labels to determine sizes (assuming same as images)
            label_df = pd.read_parquet(f)
            n = len(label_df)
            self.index_dict[i] = range(n)
            if self.label_names is None:
                self.label_names = label_df.columns.tolist()

        # Single-file cache
        self._last_file_idx = None
        self._last_images = None
        self._last_labels = None

        # make a flat index list (file_index, internal_index)
        self.index = []

        file_indices = list(self.index_dict.keys())
        if self.shuffle:
            random.shuffle(file_indices)

        for file_idx in file_indices:
            internal_indices = list(self.index_dict[file_idx])
            if self.shuffle:
                random.shuffle(internal_indices)

            self.index.extend((file_idx, inside_idx) for inside_idx in internal_indices)


    def __len__(self):
        return len(self.index)


    def _get_from_files(self, file_idx):
        # if it's the same file we used last time, reuse it
        if self._last_file_idx == file_idx:
            return self._last_images, self._last_labels
        
        image_path = self.image_paths[file_idx]
        label_path = self.label_paths[file_idx]

        self._last_images = pd.read_parquet(image_path).values
        self._last_labels = pd.read_parquet(label_path).values

        assert len(self._last_images) == len(self._last_labels), \
            f"Image and label file length mismatch ({image_path}, {label_path})"
        
        self._last_file_idx = file_idx

        return self._last_images, self._last_labels
    

    def __getitem__(self, idx):
        """
        Standard way to iterate through the dataset, returning (image, label) pairs.
        Uses the global index to determine which file and which internal index to access,
        then applies transforms if provided. Unlike usual, get_transform() takes both image
        and label as input to allow for joint transforms.
        """
        file_idx, inside_idx = self.index[idx]

        images, labels = self._get_from_files(file_idx)

        x = images[inside_idx]
        y = labels[inside_idx]

        if self.transform:
            x = self.transform(x, y)

        if self.target_transform:
            y = self.target_transform(y)

        return x, y

   
    def get_all_class_labels(self):
        """
        Helper method to access all class labels in the dataset, e.g. for calculating class distributions.
        This assumes one-hot encoding for classification (Not applicable for regression targets).
        More efficient than iterating through __getitem__ since it doesn't need to load the large inputs.
        """
        all_labels = []
        for f in self.label_paths:
            label_df = pd.read_parquet(f)
            labels = label_df.values
            if self.target_transform:
                labels = self.target_transform(labels)
            assert torch.all((labels == 0) | (labels == 1)) and torch.all(labels.sum(dim=-1) == 1), \
                "Expected one-hot encoded binary labels after target_transform for classification"
            labels = np.argmax(labels, axis=1)  # convert one-hot to class indices
            all_labels.append(labels)
        return np.concatenate(all_labels, axis=0)


if __name__ == "__main__":
    from tqdm import tqdm
    from glob import glob

    # Example usage
    image_paths = sorted(glob("data/10783560/recon3D/recon3D_d1730*.parquet"))
    label_paths = sorted(glob("data/10783560/labels/labels_d1730*.parquet"))

    reshape_x = lambda x, y: (x.reshape((20, 13, 21)), y)

    dataset = SmartPixelsDataset(image_paths, label_paths, shuffle=True, joint_transform=reshape_x)
    print(f"Total samples in dataset: {len(dataset)}")
    print(f"dataset label names: {dataset.label_names}")

    # Access a sample
    sample_image, sample_label = dataset[0]
    print(f"Sample image shape: {sample_image.shape}, Sample label: {sample_label}")

    # Iterate through the dataset
    for i in tqdm(range(len(dataset))):
        batch_images, batch_labels = dataset[i]
