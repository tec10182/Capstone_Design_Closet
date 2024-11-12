import numpy as np
from typing import List
from PIL import Image


def make_tryon_img(categories: List[str], images: List[np.ndarray]):
    """ """
    image_path = "D:/storage/test/test.png"
    image = Image.open(image_path)

    # 만약 이미지가 RGBA 모드라면 RGB로 변환
    if image.mode == "RGBA":
        image = image.convert("RGB")

    # 이미지를 numpy array로 변환
    image_array = np.array(image)

    return image_array
