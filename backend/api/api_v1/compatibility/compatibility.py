from fastapi import File, UploadFile, APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
import shutil

from sqlalchemy.orm import Session
from typing import List

from core import *
from scheme import *
from crud import *

from db import *

from PIL import Image

import io

import numpy as np

router = APIRouter()
