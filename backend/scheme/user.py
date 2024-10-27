from typing import Union, List
from pydantic import BaseModel


class User(BaseModel):
    id: str
    password: str


class Login(BaseModel):
    success: bool
