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
