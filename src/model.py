import torch
from torchvision import models


def create_model(num_classes):
    """
    Create a ResNet18 model for our vehicle classification task.
    """

    # Load pre-trained ResNet18
    model = models.resnet18(weights="DEFAULT")

    # Replace the final layer
    model.fc = torch.nn.Linear(
        model.fc.in_features,
        num_classes
    )

    return model