# 🩺 Skin Cancer Detection Using Deep Learning with Model Pruning & Quantization  



---

## 📖 Overview
Skin cancer is one of the most common and life-threatening cancers, with **melanoma** being the deadliest if undiagnosed early. Deep learning enables **dermatologist-level classification**, but models are often **too large** for **mobile and edge devices**.  

This project applies **model pruning and quantization** to CNNs trained on the **HAM10000 dataset**, reducing memory and compute costs while retaining accuracy:  

- **Pruning**: Eliminates redundant parameters  
- **Quantization**: Lowers numerical precision for faster, lighter inference  

✅ **Results:** Up to **6.5× smaller models** and **4× faster inference**, making **real-time detection feasible on mobile devices**.  

---

## 🧾 Keywords
- Skin Cancer  
- HAM10000 Dataset  
- Deep Learning  
- Model Pruning  
- Quantization  
- Model Compression  

---

## 🎯 Features
- Implement **CNN pruning & quantization** for model efficiency  
- Evaluate on **HAM10000 dermatoscopic images**  
- Benchmark **EfficientNet, ResNet, DenseNet**  
- Analyze trade-offs between **accuracy, latency, and size**  
- Deploy models on **mobile & edge devices**  
- Build a **Flask web app** for real-world usability  

---

## ⚙️ Methodology

### 1. Dataset
HAM10000 (**10,015 images**, 7 skin lesion classes):

- Actinic Keratoses  
- Basal Cell Carcinoma  
- Benign Keratosis-like Lesions  
- Dermatofibroma  
- Melanoma  
- Melanocytic Nevi  
- Vascular Lesions  

### 2. Preprocessing
- Resize → 224×224  
- Normalize (0–1 range)  
- Data augmentation: flip, rotate, scale  
- **Pruning applied**  
- Train/Validation/Test split: **80/10/10**  

### 3. CNN Models
- EfficientNetB0, EfficientNetB4  
- ResNet50  
- DenseNet169  

### 4. Compression Techniques
- **Dynamic & Static Quantization**  
- Parameter **Pruning**  

### 5. Evaluation Metrics
- **Accuracy**  
- **ROC-AUC**  
- **Cohen’s Kappa**  
- **Matthews Correlation Coefficient (MCC)**  
- **Precision, Recall, F1-Score, Support**  
- Model size (MB) & inference time (ms/sample)  
- Confusion matrix & classification report   

---

## 📊 Results Summary

| Model                | Accuracy (%) | AUC    | Kappa  | MCC    |
|---------------------|--------------|--------|--------|--------|
| EfficientNetB4      | 94.07        | 0.9953 | 0.9308 | 0.9308 |
| ResNet50            | 91.07        | 0.9904 | 0.8958 | 0.8961 |
| DenseNet169         | 92.64        | 0.9939 | 0.9141 | 0.9143 |
| EfficientNetB4 (DQ) | **98.43**    | 0.9996 | 0.9816 | 0.9816 |
| EfficientNetB0      | 92.50        | 0.9929 | 0.9125 | 0.9125 |
| EfficientNetB0 (DQ) | **98.57**    | 0.9995 | 0.9833 | 0.9833 |
| ResNet50 (DQ)       | 96.93        | 0.9980 | 0.9642 | 0.9643 |
| ResNet50 (SQ)       | 94.86        | 0.9975 | 0.9400 | 0.9400 |
| DenseNet169 (DQ)    | 97.47        | 0.9987 | 0.9700 | 0.9701 |

✅ **Dynamic Quantization (DQ)** provides the best **accuracy-latency trade-off**  
✅ **EfficientNetB0 & ResNet50 (DQ)** are ideal for **mobile deployment**  

---

## 🌐 Web Application
The Flask-based web app includes:

- 🔐 **Authentication**: Register, login, email verification, profile  
- 📷 **Detection Module**: Upload dermatoscopic images for prediction  
- ⚙️ **Pipeline**: Preprocessing → CNN inference → Confidence-based results  

**Sample Screens:** Home | Login/Register | Upload & Detect | Prediction Results  

---
---
## Project Demo Video
<a href="https://www.youtube.com/watch?v=ekwOqZq45rc" target="_blank">
  <img src="https://img.youtube.com/vi/ekwOqZq45rc/maxresdefault.jpg" alt="Watch the Demo" width="800" />
</a>

---
## 🚀 Project Setup

### 🔧 Requirements
- Python 3.8+  
- Flask, Flask-Mail, SQLite3  
- PyTorch  
- Werkzeug  

### 📥 Installation
```bash
# Clone repository
git clone https://github.com/chayon20/SkinCancerDetection
cd SkinCancerDetection

# Install dependencies
pip install -r requirements.txt

# Run the web app
python app.py
