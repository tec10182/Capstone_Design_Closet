import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

from fastapi import APIRouter, Depends, File, UploadFile, Body, Form
from fastapi.responses import JSONResponse
import shutil
import warnings

from sqlalchemy.orm import Session
from typing import List
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import AutoImageProcessor, AutoModelForImageClassification

from core import *
from scheme import *
from crud import *
from utils import *
from model import *

from db import *

from PIL import Image
import io
import numpy as np
import pickle

import time

router = APIRouter()
warnings.filterwarnings('ignore') # 짜잘한 에러 무시

#description 및 category model 불러오기
Blip_path = "C:\\Users\\han\\PycharmProjects\\Backend\\Capstone_Design_Closet\\backend\\model\\descriptor\\blip_base_fashion\\"
description_processor = BlipProcessor.from_pretrained(Blip_path)
description_model = BlipForConditionalGeneration.from_pretrained(Blip_path, from_tf=True).to("cuda")

swin_path = "C:\\Users\\han\\PycharmProjects\\Backend\\Capstone_Design_Closet\\backend\\model\\descriptor\\swin\\"
category_processor = AutoImageProcessor.from_pretrained(swin_path)
category_model = AutoModelForImageClassification.from_pretrained(swin_path).to("cuda")

#compatibility model 불러오기
args = Args()
args.model_path = './model/compatibility/src/checkpoints/cp_auc0.91.pth'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

compatibility_model, input_processor = load_model(args)
compatibility_model.to(device)
compatibility_model.eval()


# http://127.0.0.1:8000/api/v1/upload/img
@router.post("/img")
async def upload_image(
    id: str = Form(...), file: UploadFile = File(...), db: Session = Depends(get_db)
) -> Upload:
    try:
        #이미지를 업로드 해서 np.array로 변환하는 시간
        start_time = time.time()
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode == "RGBA":
            image = image.convert("RGB")
        numpy_image = np.array(image)
        print("image 가지고 와서 np.array로 변환하는 시간 :",start_time-time.time())

        # 이미지 description 생성 함수 -> model/descriptor/inference.py
        start_time = time.time()
        description = make_description(numpy_image,description_processor,description_model)
        print("이미지 description 생성하는 시간 :",start_time-time.time())

        # 이미지 description 변경 함수 -> backend/utils.py
        start_time = time.time()
        description = change_description(description)
        print("description 수정하는 시간: ",start_time-time.time())

        # 카테고리 생성함수 ->model/descriptor/inference.py
        start_time = time.time()
        category = make_category(numpy_image,category_processor,category_model)
        print("카테고리 생성하는 시간 :",start_time-time.time())

        # 이미지 임베딩 하는 함수 -> model/compatibility/inference.py
        start_time = time.time()
        embedding = make_embedding(numpy_image, description, category,compatibility_model, input_processor)
        print("이미지 임베딩 하는 시간 :",start_time-time.time())

        # 임베딩 저장하는 부분 -> backend/utils.py
        start_time = time.time()
        save_path = os.path.join(settings.storage_path, "embeddings")
        os.makedirs(save_path, exist_ok=True)
        # npy_name = make_file_name(save_path)
        # full_path = os.path.join(save_path, f"{npy_name}.npy")
        # np.save(full_path, embedding)

        #dict 형태로 저장하기 위해 pickle 저장
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
        print("모든 데이터를 로컬 폴더 및 DB에 저장하는 시간 :",start_time-time.time())

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"Image upload failed: {str(e)}")
    return upload
