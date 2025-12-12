"""3D Convolutional Neural Network for signal/background classification."""

import torch
import torch.nn as nn


class Conv3DModel(nn.Module):
    """3D CNN for classifying 3D charge distributions."""

    def __init__(
        self,
        input_shape: tuple = (1, 20, 13, 21),
        n_filters_1: int = 8,
        n_filters_2: int = 16,
        hidden_dim: int = 64,
        dropout: float = 0.0,
        output_dim: int = 1,
    ):
        """
        Args:
            input_shape: Shape of input (C, T, H, W)
            n_filters_1: Number of filters in first conv layer
            n_filters_2: Number of filters in second conv layer
            hidden_dim: Hidden dimension for FC layer
            dropout: Dropout probability
        """
        super().__init__()

        self.input_shape = input_shape

        self.conv1 = nn.Conv3d(input_shape[0], n_filters_1, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool3d(2)

        self.conv2 = nn.Conv3d(n_filters_1, n_filters_2, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool3d(2)

        # Calculate flattened size after convolutions
        # After 2 pooling layers: (T//4, H//4, W//4)
        t_out = input_shape[1] // 4
        h_out = input_shape[2] // 4
        w_out = input_shape[3] // 4
        flattened_size = n_filters_2 * t_out * h_out * w_out

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(flattened_size, hidden_dim)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(hidden_dim, output_dim)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    