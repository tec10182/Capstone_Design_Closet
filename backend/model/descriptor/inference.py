import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
from transformers import BlipProcessor, BlipForConditionalGeneration

def make_description(image: np.ndarray) -> str:
    Blip_path = "C:\\Users\\han\\PycharmProjects\\Backend\\backend\\model\\descriptor\\blip_base_fashion\\"
    processor = BlipProcessor.from_pretrained(Blip_path)
    model = BlipForConditionalGeneration.from_pretrained(Blip_path, from_tf=True).to("cuda")

    text = ""
    inputs = processor(image, text, return_tensors="pt").to("cuda")
    out = model.generate(**inputs)

    return processor.decode(out[0], skip_special_tokens=True)

