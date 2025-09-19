import os
import torch
from torchvision import models, transforms
from PIL import Image

# ---------------------------
# Base directory
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------
# Class names
# ---------------------------
class_names = [
    'Actinic_Keratoses',
    'Basal_Cell_Carcinoma',
    'Benign_Keratosis_like_Lesions',
    'Dermatofibroma',
    'Melanoma',
    'Melanocytic_Nevi',
    'Vascular_Lesions'
]

# ---------------------------
# Global model, transform, device
# ---------------------------
_model = None
_transform = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Load actual FP32 EfficientNet-B0
# ---------------------------
def load_model():
    global _model, _transform
    if _model is None:
        # Path to the FP32 model (state_dict)
        model_path = os.path.join(BASE_DIR, "skin_efficientnet_b0.pth")

        # Load EfficientNet-B0
        _model = models.efficientnet_b0(weights=None)
        _model.classifier[1] = torch.nn.Linear(_model.classifier[1].in_features, len(class_names))

        # Load trained FP32 weights
        _model.load_state_dict(torch.load(model_path, map_location=_device))
        _model.to(_device)
        _model.eval()

        # Preprocessing transforms
        from torchvision.models import EfficientNet_B0_Weights
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        _transform = transforms.Compose([
            transforms.Resize((224, 224)),  # EfficientNet-B0 input size
            transforms.ToTensor(),
            transforms.Normalize(mean=weights.transforms().mean,
                                 std=weights.transforms().std)
        ])

# ---------------------------
# Prediction function
# ---------------------------
def predict_skin_cancer(image_path):
    load_model()
    image = Image.open(image_path).convert("RGB")
    input_tensor = _transform(image).unsqueeze(0).to(_device)
    with torch.no_grad():
        outputs = _model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
    return class_names[pred.item()], conf.item() * 100
