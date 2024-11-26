from fastapi import APIRouter, Depends, File, UploadFile, Body, Form
from fastapi.responses import JSONResponse
from fastapi import HTTPException

import shutil

from sqlalchemy.orm import Session
from typing import List

from core import *
from scheme import *
from crud import *
from utils import *
from model import *

from db import *

from PIL import Image

import io

import numpy as np

import os

router = APIRouter()


# http://127.0.0.1:8000/api/v1/upload/img
@router.post("/img")
async def upload_image(
    id: str = Form(...), file: UploadFile = File(...), db: Session = Depends(get_db)
) -> Upload:
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode == "RGBA":
            image = image.convert("RGB")
        numpy_image = np.array(image)

        # 이미지 description 생성 함수
        description = make_description(numpy_image)

        # 이미지 description 변경 함수
        description = change_description(description)
        # 카테고리 생성함수

        # 이미지 임베딩 하는 함수
        category = make_category(numpy_image)

        embedding = make_embedding(numpy_image, description, category)

        # 임베딩 저장하는 부분
        save_path = os.path.join(settings.storage_path, "embeddings")
        os.makedirs(save_path, exist_ok=True)
        npy_name = make_file_name(save_path)
        full_path = os.path.join(save_path, f"{npy_name}.npy")
        np.save(full_path, embedding)

        # 이미지 저장하는 부분
        save_path = os.path.join(settings.storage_path, "images")
        os.makedirs(save_path, exist_ok=True)

        image_name = make_file_name(save_path)
        full_save_path = os.path.join(save_path, f"{image_name}.png")

        image.save(full_save_path)

        # db에 저장하는 함수
        insert_image(
            db,
            id,
            image_name + ".png",
            description,
            category,
            npy_name + ".npy",
        )

        # 리턴 지정하는 부분
        upload = Upload
        upload.success = True

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image upload failed: {str(e)}")
    return upload
