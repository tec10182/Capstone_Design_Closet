from typing import Union, List
from typing import Optional
from pydantic import BaseModel


class User(BaseModel):
    id: str
    password: str


class Login(BaseModel):
    success: bool


class Sign(BaseModel):
    success: bool
    msg: str


class ImageResponseModel(BaseModel):
    image: List[str]  # base64로 인코딩된 이미지 리스트
    category: List[str]  # 문자열 리스트
    image_id: List[str]

