import torch
import torchvision.models as models
from torch import nn

def create_model(model_name):
    """Create a PyTorch model with pretrained weights and modified for feature extraction"""
    if model_name == 'DenseNet169':
        model = models.densenet169(pretrained=True)
        model = nn.Sequential(*list(model.children())[:-1])  # Remove classification layer
    elif model_name == 'VGG19':
        model = models.vgg19(pretrained=True)
        model = nn.Sequential(*list(model.children())[:-1])  # Remove classification layer
    elif model_name == 'EfficientNetB2':
        model = models.efficientnet_b2(pretrained=True)
        model = nn.Sequential(*list(model.children())[:-1])  # Remove classification layer
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    model.eval()
    return model

def load_base_models():
    return {
        'DenseNet169': create_model('DenseNet169'),
        'VGG19': create_model('VGG19'),
        'EfficientNetB2': create_model('EfficientNetB2')
    }