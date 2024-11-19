import torch
import torch.nn as nn
from torch import Tensor


class MriClassifier(nn.Module):
    def __init__(self):
        super(MriClassifier, self).__init__()
        self.relu = nn.ReLU()

        # Convolutional layers with 3D operations
        self.conv1 = nn.Conv3d(1, 8, kernel_size=7, stride=2, padding=7 // 2, bias=False)
        self.conv2 = nn.Conv3d(8, 16, kernel_size=3, stride=1, padding=3 // 2, bias=False)

        # Batch normalization for 3D
        self.bn1 = nn.BatchNorm3d(8)
        self.bn2 = nn.BatchNorm3d(16)

        # Adaptive pooling to reduce to a fixed-size representation
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # Fully connected layer for binary classification
        self.fc = nn.Linear(16, 1)
        # zero initialization of the fully connected layer
        self.fc.weight.data.zero_()

    def forward(self, x: Tensor):
        # Pass through convolutional layers with ReLU activation and batch norm
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))

        # Adaptive pooling to reduce to a 1x1x1 feature map
        x = self.pool(x)

        # Flatten to a 1D vector for the fully connected layer
        x = torch.flatten(x, start_dim=1)

        # Final fully connected layer for binary output
        x = self.fc(x)
        return x

