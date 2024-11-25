import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
from PIL import Image
import torch
import warnings
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import AutoImageProcessor, AutoModelForImageClassification

warnings.filterwarnings("ignore")  # 짜잘한 에러 무시


def make_description(
    image: np.ndarray, description_processor, description_model
) -> str:
    text = ""
    inputs = description_processor(image, text, return_tensors="pt").to("cuda")
    out = description_model.generate(**inputs)

    return description_processor.decode(out[0], skip_special_tokens=True)


def load_description_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    Blip_path = os.path.join(current_dir, "blip_base_fashion")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    description_processor = BlipProcessor.from_pretrained(Blip_path)
    description_model = BlipForConditionalGeneration.from_pretrained(
        Blip_path, from_tf=True
    ).to(device)

    return description_model, description_processor
