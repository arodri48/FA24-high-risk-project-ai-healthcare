from torch import nn
import torch

class MultimodalClassifier(nn.Module):
    def __init__(self, num_categorical_features: int, num_classes: int):
        super(MultimodalClassifier, self).__init__()

        # Image branch: Pretrained ResNet model for image features
        self.image_branch = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=7 // 2),  # Start with larger receptive field
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),  # Optional additional conv layer
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # Global pooling to reduce output to (batch, 512, 1, 1)
            nn.Flatten()
        )

        # Categorical branch: Simple MLP for categorical features
        self.categorical_branch = nn.Sequential(
            nn.Linear(num_categorical_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

        # Combined branch: Combine both branches and add further dense layers
        self.combined_branch = nn.Sequential(
            nn.Linear(num_features + 32, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, image, categorical):
        # Forward pass through the image branch
        image_features = self.image_branch(image)  # (batch_size, num_features)

        # Forward pass through the categorical branch
        categorical_features = self.categorical_branch(categorical)  # (batch_size, 32)

        # Concatenate the features from both branches
        combined_features = torch.cat((image_features, categorical_features), dim=1)

        # Forward pass through the combined branch
        output = self.combined_branch(combined_features)
        return output