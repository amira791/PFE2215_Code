import os
import time
import torch
import numpy as np
from system.base_models import load_base_models
from system.eca_resnet import CropDiseaseModel
from system.model_utils import load_svm_model, extract_features
from data.preprocessing import preprocess_image_torch, load_image_for_torch
from evaluation.metrics import calculate_metrics, print_metrics
from evaluation.visualize import plot_confusion_matrix
from sklearn.model_selection import train_test_split

def load_dataset(dataset_path="PFE_dataset"):
    classes = ["Healthy"] + sorted([c for c in os.listdir(dataset_path) if c != "Healthy"])
    label_map = {cls: idx for idx, cls in enumerate(classes)}
    
    image_paths = []
    labels = []
    
    for cls in classes:
        cls_folder = os.path.join(dataset_path, cls)
        for fname in os.listdir(cls_folder):
            if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                image_paths.append(os.path.join(cls_folder, fname))
                labels.append(label_map[cls])
    
    return train_test_split(image_paths, labels, test_size=0.2, random_state=42, stratify=labels)

def main():
    # Load models
    base_models = load_base_models()
    svm_model = load_svm_model("M1_svm.pkl")
    
    # Load CropDiseaseModel
    model2 = CropDiseaseModel(num_classes=9)
    model2.load_state_dict(torch.load("M2_ECA.pth", map_location=torch.device("cpu")))
    model2.eval()
    
    # Load dataset
    train_paths, test_paths, train_labels, test_labels = load_dataset()
    
    # Evaluation
    y_true = []
    y_pred = []
    total_inference_time = 0.0
    
    for img_path, true_label in zip(test_paths, test_labels):
        try:
            start_time = time.time()
            
            # Preprocess for feature extraction models
            image_tensor = preprocess_image_torch(img_path)
            
            # Extract features from each model
            features = []
            for model_name, model in base_models.items():
                feat = extract_features(model, image_tensor)
                if len(feat.shape) > 1:  # If features are 2D (e.g., from CNN)
                    feat = feat.mean(axis=(1, 2))  # Global average pooling
                features.append(feat)
            
            features = np.concatenate(features, axis=0)
            features = features.reshape(1, -1)  # Reshape for SVM
            
            pred_m1 = svm_model.predict(features)[0]

            if pred_m1 == 0: 
                final_pred = label_map["Healthy"]
            else:
                input_tensor = load_image_for_torch(img_path)
                with torch.no_grad():
                    output = model2(input_tensor)
                    probs = torch.nn.functional.softmax(output, dim=1)
                    pred_label = torch.argmax(probs).item()
                    final_pred = pred_label + 1 

            end_time = time.time()  
            inference_time = end_time - start_time
            total_inference_time += inference_time

            y_true.append(true_label)
            y_pred.append(final_pred)

        except Exception as e:
            print(f"❌ Failed to process image: {img_path} | Error: {e}")

    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred)
    print_metrics(metrics)
    
    avg_time = total_inference_time / len(y_true) if y_true else 0
    print(f"\n⏱️ Total Inference Time: {total_inference_time:.4f} seconds")
    print(f"⏱️ Average Inference Time per Image: {avg_time:.4f} seconds")
    
    # Plot confusion matrix
    classes = ["Healthy"] + sorted([c for c in os.listdir("PFE_dataset") if c != "Healthy"])
    plot_confusion_matrix(metrics['confusion_matrix'], classes)

if __name__ == "__main__":
    main()