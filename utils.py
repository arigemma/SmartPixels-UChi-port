import argparse
import os
import numpy as np


class IsReadableDir(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        prospective_dir = values
        if not os.path.isdir(prospective_dir):
            raise argparse.ArgumentTypeError(
                "{0} is not a valid path".format(prospective_dir)
            )
        if os.access(prospective_dir, os.R_OK):
            setattr(namespace, self.dest, prospective_dir)
        else:
            raise argparse.ArgumentTypeError(
                "{0} is not a readable directory".format(prospective_dir)
            )


class IsValidFile(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        prospective_file = values
        if not os.path.exists(prospective_file):
            raise argparse.ArgumentTypeError(
                "{0} is not a valid file".format(prospective_file)
            )
        else:
            setattr(namespace, self.dest, prospective_file)


class CreateFolder(argparse.Action):
    """
    Custom action: create a new folder if not exist. If the folder
    already exists, do nothing.

    The action will strip off trailing slashes from the folder's name.
    """

    def create_folder(self, folder_name):
        """
        Create a new directory if not exist. The action might throw
        OSError, along with other kinds of exception
        """
        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)

        # folder_name = folder_name.rstrip(os.sep)
        folder_name = os.path.normpath(folder_name)
        return folder_name

    def __call__(self, parser, namespace, values, option_string=None):
        if type(values) == list:
            folders = list(map(self.create_folder, values))
        else:
            folders = self.create_folder(values)
        setattr(namespace, self.dest, folders)


class SymLogTransform:
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.sign(x) * np.log1p(np.abs(x))


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        """
        Args:
            patience (int): how many epochs to wait after last improvement.
            min_delta (float): minimum change in the monitored quantity to count as improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def step(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

        return self.early_stop  