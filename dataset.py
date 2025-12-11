from typing import List
from torch.utils.data import Dataset
import pandas as pd


class SmartPixelsDataset(Dataset):
    """
    PyTorch Dataset for SmartPixels 3D charge distribution data with lazy loading and single-file caching.
    Using <image> to refer to the 3D charge distribution data and <label> to refer to the associated labels.
    Make sure to provide matching lists of image and label file paths, e.g.
         [image1.parquet, image2.parquet], [label1.parquet, label2.parquet]
    Args:
        image_paths: List of image file paths
        label_paths: List of label file paths
        transform: Optional transform to be applied on a sample.
        target_transform: Optional transform to be applied on the target.
    """
    def __init__(self, image_paths: List[str], label_paths: List[str], transform=None, target_transform=None):
        assert len(image_paths) == len(label_paths), "Number of image and label files must match."

        self.image_paths = image_paths
        self.label_paths = label_paths

        self.transform = transform
        self.target_transform = target_transform

        # Build a global index: (file_index, internal_index)
        self.index = []
        self.sizes = []
        for f in self.label_paths:  # use labels to determine sizes (assuming same as images)
            label_df = pd.read_parquet(f)
            n = len(label_df)
            self.sizes.append(n)
            self.index.extend([(len(self.sizes)-1, i) for i in range(n)])

        # Single-file cache
        self._last_file_idx = None
        self._last_images = None
        self._last_labels = None


    def __len__(self):
        return len(self.index)
    

    def _get_file(self, file_idx):
        # if it's the same file we used last time, reuse it
        if self._last_file_idx == file_idx:
            return self._last_images, self._last_labels
        
        image_path = self.image_paths[file_idx]
        label_path = self.label_paths[file_idx]

        self._last_images = pd.read_parquet(image_path)
        self._last_labels = pd.read_parquet(label_path)

        assert len(self._last_images) == len(self._last_labels), \
            f"Image and label file length mismatch ({image_path}, {label_path})"
        
        self._last_file_idx = file_idx

        return self._last_images, self._last_labels


    def __getitem__(self, idx):
        file_idx, inside_idx = self.index[idx]
        images, labels = self._get_file(file_idx)

        x = images.iloc[inside_idx]
        y = labels.iloc[inside_idx]

        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)

        return x, y
        
    
if __name__ == "__main__":
    from tqdm import tqdm
    from glob import glob

    # Example usage
    image_paths = sorted(glob("data/10783560/recon3D/recon3D_d1730*.parquet"))
    label_paths = sorted(glob("data/10783560/labels/labels_d1730*.parquet"))

    dataset = SmartPixelsDataset(image_paths, label_paths)
    print(f"Total samples in dataset: {len(dataset)}")

    # Access a sample
    sample_image, sample_label = dataset[0]
    print(f"Sample image shape: {sample_image.shape}, Sample label: {sample_label}")

    # iterate through the dataset
    for i in tqdm(range(len(dataset))):
        img, lbl = dataset[i]
