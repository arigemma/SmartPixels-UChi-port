"""Model definitions for SmartPixels classification."""

from .conv3d import Conv3DModel
from .conv2d import Conv2DModel
from .sklearn_wrapper import (
    SklearnModelWrapper,
    create_random_forest,
    create_logistic_regression,
    create_svm
)

__all__ = [
    "Conv3DModel",
    "Conv2DModel",
    "SklearnModelWrapper",
    "create_random_forest",
    "create_logistic_regression",
    "create_svm",
]


def get_model(name, input_shape=(1, 20, 13, 21), output_dim=13, size=4):
    if name == "conv2d":
        model = Conv2DModel(
            input_shape=input_shape,
            output_dim=output_dim,
            n_filters_1=2*size,
            n_filters_2=4*size,
            hidden_dim=8*size,
            dropout=0.1,
            aggregation="max",
        )
        return model
    elif name == "conv3d":
        model = Conv3DModel(
            input_shape=input_shape,
            output_dim=output_dim,
            n_filters_1=size,
            n_filters_2=2*size,
            hidden_dim=4*size,
            dropout=0.1,
            # aggregation="max",
        )
        return model
