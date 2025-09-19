# 🩺 Skin Cancer Detection Using Deep Learning with Model Pruning & Quantization  

## 📖 Abstract  
Skin cancer is among the most common and life-threatening cancers, with **melanoma** being the deadliest when left undiagnosed. While deep learning has enabled dermatologist-level classification of skin lesions, models are often **too large and computationally expensive** for deployment on **mobile and edge devices**.  

This project applies **model pruning and quantization** to compress CNNs trained on the **HAM10000 dataset**, reducing memory and compute costs while retaining accuracy.  
- **Pruning** eliminates redundant parameters.  
- **Quantization** lowers numerical precision for faster, lighter inference.  

Our results show **up to 6.5× model size reduction** and **4× faster inference**, making **real-time skin cancer detection feasible on resource-constrained devices**.  

---

## 🧾 Keywords  
- Skin Cancer  
- HAM10000 Dataset  
- Deep Learning  
- Model Pruning  
- Quantization  
- Model Compression  

---

## 🎯 Objectives  
- Implement **CNN pruning & quantization** for model efficiency.  
- Evaluate performance on **HAM10000 dermatoscopic images**.  
- Benchmark **EfficientNet, ResNet, and DenseNet**.  
- Analyze trade-offs between **accuracy, latency, and size**.  
- Deploy models on **mobile & edge devices**.  
- Build a **Flask web app** for real-world usability.  



## ⚙️ Methodology  
1. **Dataset**: HAM10000 (10,015 images across 7 skin lesion classes).  
2. **Preprocessing**:  
   - Resize → 224×224  
   - Normalize (0–1 range)  
   - Augment (flip, rotate, scale)
   - **Pruning**  
   - Train/Val/Test split: **80/10/10**  
3. **CNN Models**:  
   - EfficientNetB0, EfficientNetB4  
   - ResNet50  
   - DenseNet169  
4. **Compression Techniques**:  
   - **Quantization** (dynamic & static post-training)  
5. **Evaluation Metrics**:  
   - Accuracy, ROC-AUC, Cohen’s Kappa, MCC  
   - Model size (MB), inference time (ms/sample)  
   - Confusion matrix & classification report  

---

## 📊 Results Summary  

| Model                | Accuracy (%) | AUC    | Kappa  | MCC    |
|-----------------------|--------------|--------|--------|--------|
| EfficientNetB4        | 94.07        | 0.9953 | 0.9308 | 0.9308 |
| ResNet50              | 91.07        | 0.9904 | 0.8958 | 0.8961 |
| DenseNet169           | 92.64        | 0.9939 | 0.9141 | 0.9143 |
| EfficientNetB4 (DQ)   | **98.43**    | 0.9996 | 0.9816 | 0.9816 |
| EfficientNetB0        | 92.50        | 0.9929 | 0.9125 | 0.9125 |
| EfficientNetB0 (DQ)   | **98.57**    | 0.9995 | 0.9833 | 0.9833 |
| ResNet50 (DQ)         | 96.93        | 0.9980 | 0.9642 | 0.9643 |
| ResNet50 (SQ)         | 94.86        | 0.9975 | 0.9400 | 0.9400 |
| DenseNet169 (DQ)      | 97.47        | 0.9987 | 0.9700 | 0.9701 |

  

---

## 🌐 Web Application  
The Flask-based web app includes:  
- 🔐 **Authentication**: Register, login, email verification, profile.  
- 📷 **Detection Module**: Upload dermatoscopic images for prediction.  
- ⚙️ **Pipeline**: Image preprocessing → CNN inference → Confidence-based result.  

📸 **Sample Screens**:  
- Home Page  
- Login/Register  
- Upload & Detect  
- Prediction Results  

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
git clone https://github.com/yourusername/SkinCancerDetection.git
cd SkinCancerDetection

# Install dependencies
pip install -r requirements.txt

# Run the web app
python app.py
