import torch
from .registry import register


# models like in paper: https://iopscience.iop.org/article/10.1088/2632-2153/ad6a00/pdf
# Some text extracts at the bottom of this file

@register("towards-model-1", input_type="y-size", task_type="classification")
class TowardsModel1(torch.nn.Module):
    def __init__(self):
        super(TowardsModel1, self).__init__()
        self.fc1 = torch.nn.Linear(2, 128)
        self.fc2 = torch.nn.Linear(128, 3)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        assert x.shape == (x.shape[0], 2), f"Expected input shape (batch_size, 2), but got {x.shape}"
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)


@register("towards-model-2", input_type="y-profile", task_type="classification")
class TowardsModel2(torch.nn.Module):
    def __init__(self):
        super(TowardsModel2, self).__init__()
        self.fc1 = torch.nn.Linear(14, 128)
        self.fc2 = torch.nn.Linear(128, 3)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        assert x.shape == (x.shape[0], 14), f"Expected input shape (batch_size, 14), but got {x.shape}"
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)
    

@register("towards-model-3", input_type="y-profile-timing", task_type="classification")
class TowardsModel3(torch.nn.Module):
    def __init__(self):
        super(TowardsModel3, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=3, stride=1)  # out shape (16, 11, 6)
        self.conv2 = torch.nn.Conv2d(16, 64, kernel_size=3, stride=1)  # out shape (64, 9, 4)
        self.fc1 = torch.nn.Linear(64 * 9 * 4 + 1, 32)  # Adjusted for the output size of conv layers
        self.dropout = torch.nn.Dropout(0.1)
        self.fc2 = torch.nn.Linear(32, 3)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        assert x.shape == (x.shape[0], 105), f"Expected input shape (batch_size, 105), but got {x.shape}"

        y0 = x[:, 0:1]  # Extract y0 (shape: (batch_size, 1))
        profile = x[:, 1:].view(-1, 1, 13, 8)  # Reshape to (batch_size, 1, 13, 8

        x = torch.relu(self.conv1(profile))
        x = torch.relu(self.conv2(x))
        
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.cat((y0, x), dim=1)  # Concatenate y0 with flattened conv output
        
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return self.softmax(x)
    

# @register("towards-model-4", input_type="y-profile-timing", task_type="classification")    

""" from the original papers
(i) Model 1: cluster y-size. This model used the cluster position on a flat module (y0), and cluster y-size,
which is the number of pixel rows with nonzero charge deposited after 4 nanoseconds. This model has
two input features: y0 position (1 feature) and cluster y-size (1 feature). This model consists of one
dense layer with 128 neurons and 384 parameters, followed by one dense layer with 3 neurons and 387
parameters. The model provides a test of performance with minimal information provided to the
neural network.
(ii) Model 2: cluster y-profile. This model has fourteen input features: y0 position (1 feature) and cluster
y-profile (13 features corresponding to 13 pixel rows). Cluster y-profile represents the amount of
charge collected in each row of pixels after 4 nanoseconds. The model consists of one dense layer with
128 neurons and 1920 parameters, followed by one dense layer with 3 neurons and 387 parameters.
(iii) Model 3: cluster y-profile with timing information. The third and most complex model takes as input
the y0 position (1 feature) and the cluster y-profile distribution at eight time slices (13 × 8 features),
which represents the amount of charge collected in each row of pixels evaluated at eight intervals of
200 picoseconds. The earliest time slices contain the most useful information, as most charge
deposition occurs at the beginning of the cluster time evolution. This model uses a convolutional
neural network to pass a time-lapse picture of the cluster charge to the network. Cluster y-profile inputs
were passed through two two-dimensional convolutional layers (Conv2D), with 16 and 64 filters,
respectively, using ReLU activations to introduce non-linearity [23]. The shape of the kernels was 3 × 3,
and strides was 1 × 1. The output of the Conv2D layers was flattened and concatenated with the y0
input. This was then passed through a dense layer with 32 neurons, using dropout of 0.1. The final
model contains 83 331 parameters.

The cluster y-profile is calculated
for each event as the sensor output at 4000 ps summed over pixel rows (x) to project the integrated charge
of the cluster onto the y-axis. Additionally, y0, the azimuthal position of the particle’s incident position on
the sensor array in the global detector coordinates, is passed as input. The inter-dependence of the cluster y-
profile, y0, and pT was examined in [1]. This network, therefore taking in 14 input values (y0 and 13 values
from cluster y-profile), contains one dense hidden layer with 128 neurons and 1920 parameters, followed
by one dense output layer with 3 neurons and 387 parameters. A softmax activation is used to generate
classification probabilities between 0 and 1, and each event is assigned the classification label corresponding
to the highest probability.
"""