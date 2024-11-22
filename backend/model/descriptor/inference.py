import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

import numpy as np
from PIL import Image
import torch
import warnings
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import AutoImageProcessor, AutoModelForImageClassification
warnings.filterwarnings('ignore') # 짜잘한 에러 무시


def make_description(image: np.ndarray,description_processor,description_model) -> str:
    # Blip_path = "C:\\Users\\han\\PycharmProjects\\Backend\\Capstone_Design_Closet\\backend\\model\\descriptor\\blip_base_fashion\\"
    # processor = BlipProcessor.from_pretrained(Blip_path)
    # model = BlipForConditionalGeneration.from_pretrained(Blip_path, from_tf=True).to("cuda")

    text = ""
    inputs = description_processor(image, text, return_tensors="pt").to("cuda")
    out = description_model.generate(**inputs)

    return description_processor.decode(out[0], skip_special_tokens=True)

def make_category(image: np.ndarray,category_processor,category_model) -> str:
    # swin_path = "C:\\Users\\han\\PycharmProjects\\Backend\\Capstone_Design_Closet\\backend\\model\\descriptor\\swin\\"
    # processor = AutoImageProcessor.from_pretrained(swin_path)
    # model = AutoModelForImageClassification.from_pretrained(swin_path).to("cuda")

    label2category = {"LABEL_0": "bottoms", "LABEL_1": "outers", "LABEL_2": "shoes_wrong category", "LABEL_3": "tops"}

    pil_image = Image.fromarray(image).convert('RGB')
    inputs = category_processor(pil_image, return_tensors="pt").to("cuda")

    with torch.no_grad():
        logits = category_model(**inputs).logits

    predicted_label = logits.argmax(-1).item()

    return label2category[category_model.config.id2label[predicted_label]]
    # return random.choice(["tops", "bottoms"])