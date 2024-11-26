from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import JSONResponse


from sqlalchemy.orm import Session
from typing import List
import platform
import os
from io import BytesIO


from scheme import *
from crud import *
from utils import *

from db import *

router = APIRouter()


# http://127.0.0.1:8000/api/v1/user/login
@router.post("/login", response_model=Login)
async def login(user: User, db: Session = Depends(get_db)) -> Login:
    """login 정보가 일치하는지 확인

    Args:
        db(Session): DB
        user(User): id(str), password(str)

    Return:
        login(Login): success(boolean)
    """
    try:
        user = read_id_password(db, user.id, user.password)
        login = Login
        if user == None:
            login.success = False
        else:
            login.success = True
    except Exception as e:
        return HTTPException(status_code=500, detail="DB ERROR")
    return login


# http://127.0.0.1:8000/api/v1/user/signup
# 문제점 db error 나오면 그냥 코드 진행됨;
@router.post("/signup", response_model=Sign)
async def signup(user: User, db: Session = Depends(get_db)) -> Sign:
    """회원가입에 대한 정보 확인

    Args:
        db(Session): DB
        user(User): id(str), password(str)

    Return:
        Sign(Sign): msg(str), success(boolean)
    """
    try:
        sign = Sign

        id_check = read_id(db, user.id)
        if id_check != None:
            sign.success = False
            sign.msg = "중복된 id 입니다"
            return sign

        password_check = read_password(db, user.password)
        if password_check != None:
            sign.success = False
            sign.msg = "중복된 password 입니다"
            return sign

        sign_check = write_user(db, user.id, user.password)
        sign.success = True
        sign.msg = "회원가입 성공"

    except Exception as e:
        return HTTPException(status_code=500, detail="DB ERROR")

    return sign


# http://127.0.0.1:8000/api/v1/user/resign
@router.post("/resign", response_model=Sign)
async def resign(user: User, db: Session = Depends(get_db)) -> Sign:
    """id password가 주어지면 유저테이블에서 유저 삭제, 관련된 파일 삭제, 옷 db에서 유저관련된 옷 삭제

    Args:
        db (Session): DB
        user (User): id(str), password(str)

    Return:
        Sign(Basemodel): msg(str), success(boolean)
    * 유저테이블에서 유저삭제만 구현됨
    """
    try:
        sign = Sign

        id_check = read_id(db, user.id)
        if id_check == None:
            sign.success = False
            sign.msg = "잘못된 id 입니다"
            return sign

        password_check = read_password(db, user.password)
        if password_check == None:
            sign.success = False
            sign.msg = "잘못된 password 입니다"
            return sign

        resign_check = erase_user(db, user.id, user.password)
        sign.success = True
        sign.msg = "회원 탈퇴 성공"

    except Exception as e:
        return HTTPException(status_code=500, detail="DB ERROR")

    return sign


# http://127.0.0.1:8000/api/v1/user/image
@router.post("/image", response_model=ImageResponseModel)
async def get_image(id: str = Query(...), db: Session = Depends(get_db)):
    result = read_image_from_id(db, id)
    image_paths = result.get("image", [])
    category = result.get("category", [])
    images = []
    for image_path in image_paths:
        path = os.path.join(settings.storage_path, "images")
        image_path = os.path.join(path, image_path)
        images.append(image_path_to_bytes(image_path))

    return ImageResponseModel(image=images, category=category, image_id=image_paths)
