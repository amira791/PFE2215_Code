# 🌾 Deep Learning for Wheat Health Monitoring – Project Repository

This repository contains a complete pipeline for detecting early-stage wheat issues, including disease and insect pest detection, based on a combination of deep learning, vegetation indices, and state-of-the-art research implementations.

---

## 📁 Project Structure

### 1. `CODE_Early_Wheat_Issues_Detection/`
This folder contains the code used in the initial stage of the project. It includes:
- Scripts for generating datasets using **vegetation indices**, particularly **NDVI (Normalized Difference Vegetation Index)**.
- Preprocessing and augmentation scripts.
- Training code for an initial classification model trained on NDVI-based data to detect general wheat issues.

---

### 2. `CODE_State_Of_The_Art/`
This directory contains the **implementation of three scientific papers** related to wheat disease and pest detection. For each paper:
- The **code is fully implemented** as described in the original works.
- Each subfolder includes the **corresponding PDF** of the paper.
- All training parameters and results are carefully reproduced and documented.
- Pretrained models and output results are shared.

---

### 3. `CODE_Wheat_Disease_Classification/`
This module focuses on identifying wheat diseases from field images. It includes:
- A **hierarchical architecture** composed of two models:
  - **Model M01**: A **binary classification model** that classifies an image as either **Healthy** or **Diseased**. It combines **pretrained CNNs** (feature extraction) with a **machine learning classifier**.
  - **Model M02**: A **multi-class classification model** based on **ResNet50** enhanced with an **ECA (Efficient Channel Attention)** module for fine-grained disease classification.
- The 9 disease classes:
  - Fusarium
  - Loose Smut
  - Yellow Rust
  - Brown Rust
  - Stem Rust
  - Tan Spot
  - Common Root Rot
  - Septoria
  - Mildew
- Plus 1 **Healthy** class.
- Full code, training scripts, and results are included.

---

### 4. `CODE_Wheat_Insect_Pest_Detection/`
This folder contains an advanced object detection model using **YOLOv8m-OBB** (Oriented Bounding Boxes) for detecting wheat pest insects. The model is trained on a custom dataset with **7 classes** of pests:
- Armyworm Moth
- Armyworm Larvae
- Aphid
- Grasshopper
- Mite
- Beetle
- Sawfly

Includes:
- Custom YOLOv8m-OBB configuration
- Annotated dataset
- Training and evaluation scripts
- Model weights and prediction examples

---

## 🧠 Technologies Used
- Python, PyTorch, OpenCV
- YOLOv8 + Oriented Bounding Boxes (OBB)
- CNNs (ResNet, EfficientNet, etc.)
- Machine Learning classifiers (SVM, Random Forest)
- NDVI and vegetation index-based analysis
- Satellite/drone imagery preprocessing

---

## 📈 Results
Each module includes:
- Confusion matrices
- Accuracy/F1-score reports
- Trained models
- Visualizations and prediction examples

---



## ✍️ Authors
Bellali Amira, Abbaci Zoulikha – Final Year Project Students, ESI – 2024/2025

---

### 👨‍🏫 Supervisors

- **Dr. Abdenour Sehad** – ESI, Algiers  
- **Dr. Naima Bessah** – ESI, Algiers  
- **Dr. Rachid Hedjam** – Bishop’s University, Sherbrooke, Canada


