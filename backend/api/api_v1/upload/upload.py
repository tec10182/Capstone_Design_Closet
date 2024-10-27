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


@router.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        numpy_image = np.array(image)

        image_name = "tmp"

        image.save(settings.storage_path, image_name)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image upload failed: {str(e)}")
