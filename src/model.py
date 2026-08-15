import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def get_model(architecture: str = "resnet18", num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    """
    Constructs and returns image classification model architecture.
    """
    if architecture.lower() == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
