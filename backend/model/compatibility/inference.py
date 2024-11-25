import numpy as np
import random

def make_embedding(
    image: np.ndarray, description: str, category: str
):
    return np.random.rand(128)

def make_score(embedding:np.ndarray):
    return random.randint(1,100)