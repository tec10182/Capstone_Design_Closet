import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from fastapi import APIRouter, Depends, File, UploadFile, Body, Form
from fastapi.responses import JSONResponse
import shutil
import warnings

from sqlalchemy.orm import Session
from typing import List
import torch

from core import *
from scheme import *
from crud import *
from utils import *
from model import *

from db import *

from PIL import Image, ImageOps
import io
import numpy as np
import pickle

import time

router = APIRouter()
warnings.filterwarnings("ignore")  # 짜잘한 에러 무시

description_model, description_processor = load_description_model()
category_model, category_processor = load_classification_model()
compatibility_model, input_processor = load_compatibility_model()
compatibility_model.eval()


# http://127.0.0.1:8000/api/v1/upload/img
@router.post("/img")
async def upload_image(
    id: str = Form(...), file: UploadFile = File(...), db: Session = Depends(get_db)
) -> Upload:
    try:

        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        image = ImageOps.exif_transpose(image)
        if image.mode == "RGBA":
            image = image.convert("RGB")
        numpy_image = np.array(image)

        description = make_description(
            numpy_image, description_processor, description_model
        )

        description = change_description(description)

        category = make_category(numpy_image, category_processor, category_model)

        embedding = make_embedding(
            numpy_image, description, category, compatibility_model, input_processor
        )

        # 임베딩 저장하는 부분 -> backend/utils.py
        save_path = os.path.join(settings.storage_path, "embeddings")
        os.makedirs(save_path, exist_ok=True)

        # dict 형태로 저장하기 위해 pickle 저장
        pkl_name = make_file_name(save_path)
        with open(os.path.join(save_path, f"{pkl_name}.pickle"), "wb") as f:
            pickle.dump(embedding, f)

        # 이미지 저장하는 부분
        save_path = os.path.join(settings.storage_path, "images")
        os.makedirs(save_path, exist_ok=True)

        image_name = make_file_name(save_path)
        full_save_path = os.path.join(save_path, f"{image_name}.png")

        image.save(full_save_path)

        # db에 저장하는 함수 -> crud/upload.py
        insert_image(
            db,
            id,
            image_name + ".png",
            description,
            category,
            pkl_name + ".pickle",
        )

        # 리턴 지정하는 부분
        upload = Upload
        upload.success = True

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"Image upload failed: {str(e)}")
    return upload
