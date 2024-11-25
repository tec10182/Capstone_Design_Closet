from fastapi import File, UploadFile, APIRouter, HTTPException, Depends, Form
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse

from sqlalchemy.orm import Session
from typing import List

from core import *
from scheme import *
from crud import *
from utils import *
from model import *

from db import *

from PIL import Image
import os
import numpy as np

from io import BytesIO

import numpy as np
import pickle

router = APIRouter()


# http://127.0.0.1:8000/api/v1/compatibility/recommend
@router.post("/recommend", response_model=CompatibilityResponseModel)
async def score(
    id: str = Form(...), file: UploadFile = File(...), db: Session = Depends(get_db)
):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode == "RGBA":
        image = image.convert("RGB")
    numpy_image = np.array(image)
    description = make_description(numpy_image)
    description = change_description(description)
    category = make_category(numpy_image)
    embedding = make_embedding(numpy_image, description, category)

    # category에 맞는 임베딩 가져오기
    if category == "tops":
        result = read_image_embedding(db, "bottoms", id)
    else:
        result = read_image_embedding(db, "tops", id)

    images = result.get("images", [])
    embeddings = result.get("embeddings", [])

    if len(embeddings) == 0:
        return CompatibilityResponseModel(
            image=[],
            score=[],
            avg_score=-1,
            success=False,
        )

    image_score_pairs = []

    anchor = embedding

    # 각 이미지와 임베딩을 매칭하여 score 계산
    for image_path, embedding_path in zip(images, embeddings):
        path = os.path.join(settings.storage_path, "images")
        image_path = os.path.join(path, image_path)

        path = os.path.join(settings.storage_path, "embeddings")
        embedding_path = os.path.join(path, embedding_path)
        embedding = np.load(embedding_path, allow_pickle=True)

        if category == "top":
            score = make_score([anchor, embedding])
        else:
            score = make_score([embedding, anchor])
        image_score_pairs.append((image_path, score))  # 이미지와 score 매칭

    # score 내림차순으로 정렬
    image_score_pairs.sort(key=lambda x: x[1], reverse=True)

    # 가장 높은 3개의 score 구하기
    best_images = [pair[0] for pair in image_score_pairs[:3]]  # 상위 3개 이미지
    best_scores = [pair[1] for pair in image_score_pairs[:3]]  # 상위 3개 score

    # 3개 score 평균 계산
    avg_score = sum(best_scores) / len(best_scores)

    # best_images를 바이너리 스트림으로 변환하여 반환
    best_images_bytes = [image_path_to_bytes(img) for img in best_images]

    # 이미지 응답 반환
    return CompatibilityResponseModel(
        image1=best_images_bytes,
        image2=best_images_bytes,
        score=best_scores,
        avg_score=int(avg_score),
        success=True,
    )


# http://127.0.0.1:8000/api/v1/compatibility/score
@router.post("/score", response_model=ScoreResponseModel)
async def single_score(image_info: Imageid,db: Session = Depends(get_db)):
    image_ids = image_info.image
    embeddings = []
    for image_id in image_ids:
        embedding_name = image_id.split(".")[0]+".npy"
        embeddings.append(embedding_name)
    score = make_score(embeddings)

    return ScoreResponseModel(
        score = score
    )

