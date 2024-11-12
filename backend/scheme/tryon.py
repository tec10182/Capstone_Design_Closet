from typing import Union, List
from typing import Optional
from pydantic import BaseModel


class Clothes(BaseModel):
    ids: List[str]
    categories: List[str]
