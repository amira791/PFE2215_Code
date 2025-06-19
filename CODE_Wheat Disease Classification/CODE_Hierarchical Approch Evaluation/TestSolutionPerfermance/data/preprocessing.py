import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

def preprocess_image_torch(image_path, img_size=(224, 224)):
    """Preprocess image for PyTorch models"""
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)  # Add batch dimension

transform_torch = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.CenterCrop((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def load_image_for_torch(image_path):
    return transform_torch(Image.open(image_path).convert('RGB')).unsqueeze(0)