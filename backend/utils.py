import os
from PIL import Image
import base64
import io
import random
from transformers import pipeline
import numpy as np


def change_description(text: str) -> str:
    """description을 compatibility 모델에 맞는 단어로 변경하는 과정

    Args:
        text(str): description

    Return:
        description(str): compatibility model에 맞는 언어
    """
    if "hanging on a wall" in text:
        text = text.replace(" hanging on a wall", "")
    if "hanging on a white wall" in text:
        text = text.replace(" hanging on a white wall", "")
    if "hanging on a door" in text:
        text = text.replace(" hanging on a door", "")
    if "hanging on a hanger" in text:
        text = text.replace(" hanging on a hanger", "")
    return text


def make_category(image: np.ndarray) -> str:
    classifier = pipeline("zero-shot-image-classification", model="patrickjohncyh/fashion-clip", device='cuda')
    classification = ['bottoms', 'tops', 'outers']

    pil_image = Image.fromarray(image)

    scores = classifier(pil_image, candidate_labels=classification)

    # 오류로 tops, bottom을 구별을 잘 못해서 임시로 랜덤으로 카테고리 넣기
    return scores[0]['label']
    # return random.choice(["tops", "bottoms"])

def make_file_name(save_path: str) -> str:
    existing_files = [
        int(f.split(".")[0]) for f in os.listdir(save_path) if f.split(".")[0].isdigit()
    ]
    max_number = max(existing_files, default=0)  # 파일이 없으면 기본값 0
    image_name = str(max_number + 1)

    return image_name


def image_path_to_bytes(image_path: str):
    with Image.open(image_path) as img:
        # RGBA 이미지일 경우 RGB로 변환
        if img.mode == "RGBA":
            img = img.convert("RGB")

        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")  # 이제 JPEG로 저장 가능
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return img_str
