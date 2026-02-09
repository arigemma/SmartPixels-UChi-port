"""2D Convolutional Neural Network"""

import torch
import torch.nn as nn
from .registry import register


@register("conv2d", input_type="full", task_type="regression")
class Conv2DModel(nn.Module):
    """2D CNN that processes each time slice and aggregates."""

    def __init__(
        self,
        input_shape: tuple = (20, 13, 21),
        output_dim: int = 13,
        n_filters_1: int = 16,
        n_filters_2: int = 32,
        hidden_dim: int = 64,
        dropout: float = 0.0,
        aggregation: str = "max"  # "max", "mean", or "flatten"
    ):
        """
        Args:
            input_shape: Shape of input (C, T, H, W)
            n_filters_1: Number of filters in first conv layer
            n_filters_2: Number of filters in second conv layer
            hidden_dim: Hidden dimension for FC layer
            dropout: Dropout probability
            aggregation: How to aggregate across time dimension
        """
        super().__init__()

        self.input_shape = input_shape
        self.output_dim = output_dim
        self.aggregation = aggregation

        # 2D convolutions operate on (H, W)
        self.conv1 = nn.Conv2d(input_shape[0], n_filters_1, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(n_filters_1, n_filters_2, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        # Calculate spatial dimensions after pooling
        h_out = input_shape[2] // 4
        w_out = input_shape[3] // 4
        feature_size = n_filters_2 * h_out * w_out

        if aggregation == "flatten":
            # Flatten across time dimension
            fc_input_size = feature_size * input_shape[1]
        else:
            # Max or mean pooling across time
            fc_input_size = feature_size

        self.fc1 = nn.Linear(fc_input_size, hidden_dim)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(hidden_dim, self.output_dim)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, T, H, W)

        Returns:
            Output predictions of shape (B, 1)
        """
        B, C, T, H, W = x.shape

        # Process each time slice through conv layers
        # Reshape to (B*T, C, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (B, T, C, H, W)
        x = x.view(B * T, C, H, W)  # (B*T, C, H, W)

        # Apply 2D convolutions
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))

        # Reshape back to (B, T, features)
        x = x.view(B, T, -1)  # (B, T, features)

        # Aggregate across time dimension
        if self.aggregation == "max":
            x = torch.max(x, dim=1)[0]  # (B, features)
        elif self.aggregation == "mean":
            x = torch.mean(x, dim=1)  # (B, features)
        elif self.aggregation == "flatten":
            x = x.view(B, -1)  # (B, T*features)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        # Fully connected layers
        x = self.relu3(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
