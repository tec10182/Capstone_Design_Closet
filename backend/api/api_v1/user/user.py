from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse


from sqlalchemy.orm import Session
from typing import List
import platform

from scheme import *
from crud import *

from db import *

router = APIRouter()


# http://127.0.0.1:8000/api/v1/user/login
@router.post("/login", response_model=Login)
async def login(user: User, db: Session = Depends(get_db)) -> Login:
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
