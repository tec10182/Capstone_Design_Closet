from typing import Union, List
from typing import Optional
from pydantic import BaseModel


class Upload(BaseModel):
    success: bool


class Id(BaseModel):
    id: str
    