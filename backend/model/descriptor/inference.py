import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
from transformers import BlipProcessor, BlipForConditionalGeneration

def make_description(image: np.ndarray) -> str:
    return "1"

