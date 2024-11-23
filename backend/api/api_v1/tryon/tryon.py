from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse


from sqlalchemy.orm import Session
from typing import List
import platform
import os
from PIL import Image
from io import BytesIO


from model import *
from scheme import *
from crud import *

from db import *

router = APIRouter()


# http://127.0.0.1:8000/api/v1/tryon/try
@router.post("/try")
async def tryon(clothes: Clothes, db: Session = Depends(get_db)):
    images = []
    for id in clothes.ids:
        image_path = os.path.join(settings.storage_path, "images")
        image_path = os.path.join(image_path, id)
        image = Image.open(image_path)
        images.append(np.array(image))

    image = make_tryon_img(clothes.categories, images)
    pil_image = Image.fromarray(image.astype("uint8"))

    # 이미지를 BytesIO 객체로 저장
    img_io = BytesIO()
    pil_image.save(img_io, "JPEG")
    img_io.seek(0)

    # FileResponse를 사용하여 메모리 내 이미지를 반환
    return StreamingResponse(img_io, media_type="image/jpeg")
