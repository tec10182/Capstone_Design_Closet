from typing import Union, List
from typing import Optional
from pydantic import BaseModel


class Cloth(BaseModel):
    category: str
    id: str
    user: str


class CompatibilityResponseModel(BaseModel):
    image1: List[str]
    image2: List[str]
    score: List[int]
    avg_score: int
    success: bool
