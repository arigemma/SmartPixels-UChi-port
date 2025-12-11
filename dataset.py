import random
from typing import List
from torch.utils.data import Dataset
import pandas as pd


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
    def __init__(self, image_paths: List[str], label_paths: List[str], shuffle: bool=False, transform=None, target_transform=None):
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
    

    def _get_file(self, file_idx):
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
        file_idx, inside_idx = self.index[idx]
        images, labels = self._get_file(file_idx)

        x = images[inside_idx]
        y = labels[inside_idx]

        if self.transform:
            x = self.transform(x)

        if self.target_transform:
            y = self.target_transform(y)

        return x, y
            
    
if __name__ == "__main__":
    from tqdm import tqdm
    from glob import glob
    from torch.utils.data import DataLoader

    # Example usage
    image_paths = sorted(glob("data/10783560/recon3D/recon3D_d1730*.parquet"))
    label_paths = sorted(glob("data/10783560/labels/labels_d1730*.parquet"))

    reshape = lambda x: x.reshape((20, 13, 21))

    dataset = SmartPixelsDataset(image_paths, label_paths, shuffle=True, transform=reshape)
    print(f"Total samples in dataset: {len(dataset)}")

    print(f"dataset label names: {dataset.label_names}")

    # Access a sample
    sample_image, sample_label = dataset[0]
    print(f"Sample image shape: {sample_image.shape}, Sample label: {sample_label}")

    # Iterate through the dataset using DataLoader
    loader = DataLoader(dataset, batch_size=128, num_workers=0, pin_memory=True)
    for batch_images, batch_labels in tqdm(loader):
        pass
