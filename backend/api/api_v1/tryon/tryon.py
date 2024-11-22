from fastapi import APIRouter, HTTPException, Depends,UploadFile, File, Form
from fastapi.responses import StreamingResponse


from sqlalchemy.orm import Session
from typing import List
import platform
import os
from PIL import Image
from io import BytesIO
import io
import uuid
import os.path as osp


from model import *
from scheme import *
from crud import *
from sympy.integrals.meijerint_doc import category

from db import *

from model.tryon.masking.model import predict_mask
from model.tryon.schp.model import human_parsing
from model.tryon.densepose.model import predict_dense_pose_map
from model.tryon.pose_estimation.model import predict_pose_kpts
from model.tryon.hr_viton.model import hr_inference
#from model.tryon.ladi_viton.model import ladi_inference

router = APIRouter()

# 기본 설정값
configs = {
    "storage": "D:\\storage",
    "available_model": {"viton": ["hr_viton", "ladi_viton"]}
}

# http://127.0.0.1:8000/api/v1/tryon/try
# @router.post("/try")
# async def tryon(clothes: Clothes, db: Session = Depends(get_db)):
#     images = []
#     for id in clothes.ids:
#         image_path = os.path.join(settings.storage_path, "images")
#         image_path = os.path.join(image_path, id)
#         image = Image.open(image_path)
#         images.append(np.array(image))
#
#     image = make_tryon_img(clothes.categories, images)
#     pil_image = Image.fromarray(image.astype("uint8"))
#
#     # 이미지를 BytesIO 객체로 저장
#     img_io = BytesIO()
#     pil_image.save(img_io, "JPEG")
#     img_io.seek(0)
#
#     # FileResponse를 사용하여 메모리 내 이미지를 반환
#     return StreamingResponse(img_io, media_type="image/jpeg")

# http://127.0.0.1:8000/api/v1/tryon/save
@router.post("/save")
async def save_images(img: UploadFile = File(...), category: str = Form(...)) -> dict:
    if category == 'cloth':
        file_content = await img.read()
        im = Image.open(io.BytesIO(file_content))
    elif category == 'person':
        im = Image.open(img.file)
    else:
        print('Available mode: cloth, person')
        raise

    im_name = str(uuid.uuid4()) + '.png'
    im_path = osp.join(configs['storage'], f'tryon/raw_data/{category}', im_name)
    im.save(im_path, 'PNG')

    save_state = False
    if osp.exists(im_path) and osp.getsize(im_path):
        save_state = True

    return {"save_state": save_state, "im_name": im_name}

@router.post("/preprocess/person")
async def pp_person(img_name: str = Form(...)) -> dict:
    """서버의 로컬 스토리지에 저장되어있는 사람 이미지의 이름을 바탕으로
    생성에 필요한 모든 전처리 과정을 수행하는 API입니다.

    Args:
        img_name (str, optional): 사람 이미지의 이름입니다. Defaults to Form(...)

    Returns:
        dict : 모든 전처리 과정에 대한 상태가 담긴 dictionary입니다.
    """
    parse_map_save_state = human_parsing(configs['storage'], img_name)
    dense_pose_save_state = predict_dense_pose_map(configs['storage'], img_name)

    mask_save_state = predict_mask(configs['storage'], img_name, mode='person')

    pose_save_state = predict_pose_kpts(configs['storage'], img_name)

    return {"parse map": parse_map_save_state, "dense pose map": dense_pose_save_state,
            "mask": mask_save_state, "pose img & kpts": pose_save_state}

@router.post("/preprocess/cloth")
async def pp_cloth(img_name: str = Form(...)) -> bool:

    mode = 'cloth'
    save_state = predict_mask(configs['storage'], img_name, mode=mode)

    return save_state


@router.post("/generate")
async def gen_viton_img(p_img_name: str = Form(...), c_img_name: str = Form(...))->str:
    img_name = hr_inference(configs['storage'], p_img_name, c_img_name)
    return img_name
    # category = "upper_body"
    # tryon_img_base64 = ladi_inference(configs['storage'], p_img_name, c_img_name, category)
    #
    # return tryon_img_base64
