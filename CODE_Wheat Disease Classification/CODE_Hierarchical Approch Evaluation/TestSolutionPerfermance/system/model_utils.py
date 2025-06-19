import joblib
import torch
import numpy as np

def load_svm_model(model_path):
    return joblib.load(model_path)

def extract_features(model, image_tensor):
    """Extract features using PyTorch model"""
    with torch.no_grad():
        features = model(image_tensor)
        return features.squeeze().cpu().numpy()