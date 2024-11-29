from fastapi import File, UploadFile, APIRouter, HTTPException, Depends, Form
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse

import warnings
from sqlalchemy.orm import Session
from typing import List
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

from core import *
from scheme import *
from crud import *
from utils import *

from db import *

from PIL import Image
import os
import numpy as np

import io

import numpy as np
import pickle

router = APIRouter()
warnings.filterwarnings("ignore")  # 짜잘한 에러 무시

description_model, description_processor = load_description_model()
category_model, category_processor = load_classification_model()
compatibility_model, input_processor = load_compatibility_model()
compatibility_model.eval()


# http://127.0.0.1:8000/api/v1/compatibility/score
@router.post("/recommend", response_model=CompatibilityResponseModel)
async def score(
    id: str = Form(...), file: UploadFile = File(...), db: Session = Depends(get_db)
):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
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

    # category에 맞는 임베딩 가져오기
    if category == "tops":
        result1 = read_image_embedding(db, "bottoms", id)
        result2 = read_image_embedding(db, "shoes", id)
    elif category == "bottoms":
        result1 = read_image_embedding(db, "tops", id)
        result2 = read_image_embedding(db, "shoes", id)

    images1 = result1.get("images", [])
    embeddings1 = result1.get("embeddings", [])

    images2 = result2.get("images", [])
    embeddings2 = result2.get("embeddings", [])

    if len(embeddings1) == 0 or len(embeddings2) == 0:
        return CompatibilityResponseModel(
            image=[],
            score=[],
            avg_score=-1,
            success=False,
        )

    image_score_pairs = []

    anchor = embedding

    # 각 이미지와 임베딩을 매칭하여 score 계산
    for image_path1, embedding_path1 in zip(images1, embeddings1):
        path = os.path.join(settings.storage_path, "images")
        image_path1 = os.path.join(path, image_path1)

        path1 = os.path.join(settings.storage_path, "embeddings")
        embedding_path1 = os.path.join(path1, embedding_path1)

        with open(embedding_path1, "rb") as fb:
            embedding1 = pickle.loads(fb.read())

        for image_path2, embedding_path2 in zip(images2, embeddings2):
            image_path2 = os.path.join(path, image_path2)

            path2 = os.path.join(settings.storage_path, "embeddings")
            embedding_path2 = os.path.join(path2, embedding_path2)

            with open(embedding_path2, "rb") as fb:
                embedding2 = pickle.loads(fb.read())

            if category == "tops":
                score = make_score(
                    [anchor, embedding1, embedding2],
                    compatibility_model,
                )
            elif category == "bottoms":
                score = make_score(
                    [embedding1, anchor, embedding2],
                    compatibility_model,
                )
            image_score_pairs.append((image_path1, image_path2, score))

    # score 내림차순으로 정렬
    image_score_pairs.sort(key=lambda x: x[2], reverse=True)

    # 가장 높은 3개의 score 구하기
    best_images = [[pair[0], pair[1]] for pair in image_score_pairs[:3]]
    best_scores = [pair[2] for pair in image_score_pairs[:3]]

    # 3개 score 평균 계산
    avg_score = sum(best_scores) / len(best_scores)

    # best_images를 바이너리 스트림으로 변환하여 반환
    best_images_bytes1 = [image_path_to_bytes(img[0]) for img in best_images]
    best_images_bytes2 = [image_path_to_bytes(img[1]) for img in best_images]

    # 이미지 응답 반환
    return CompatibilityResponseModel(
        image1=best_images_bytes1,
        image2=best_images_bytes2,
        score=best_scores,
        avg_score=int(avg_score),
        success=True,
    )


@router.post("/score", response_model=ScoreResponseModel)
async def single_score(image_info: Imageid, db: Session = Depends(get_db)):
    image_ids = image_info.image
    embeddings = []
    for image_id in image_ids:
        embedding_name = image_id.split(".")[0] + ".pickle"
        embedding_path = os.path.join(
            settings.storage_path, "embeddings", embedding_name
        )
        with open(embedding_path, "rb") as fb:
            embedding = pickle.loads(fb.read())
        embeddings.append(embedding)
    score = make_score(embeddings, compatibility_model)

    return ScoreResponseModel(score=score)
