from transformers import AutoImageProcessor, AutoModelForImageClassification
import os
import torch
from PIL import Image
import numpy as np


def load_classification_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    swin_path = os.path.join(current_dir, "swin")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    category_processor = AutoImageProcessor.from_pretrained(swin_path)
    category_model = AutoModelForImageClassification.from_pretrained(swin_path).to(
        device
    )
    return category_model, category_processor


def make_category(image: np.ndarray, category_processor, category_model) -> str:
    label2category = {
        "LABEL_0": "bottoms",
        "LABEL_1": "outers",
        "LABEL_2": "shoes",
        "LABEL_3": "tops",
    }

    pil_image = Image.fromarray(image).convert("RGB")
    inputs = category_processor(pil_image, return_tensors="pt").to("cuda")

    with torch.no_grad():
        logits = category_model(**inputs).logits

    predicted_label = logits.argmax(-1).item()
    if predicted_label == 1:
        predicted_label = 3
    return label2category[category_model.config.id2label[predicted_label]]
