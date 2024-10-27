from fastapi import APIRouter, HTTPException, Depends

from sqlalchemy.orm import Session
from typing import List
import platform

from scheme import *
from crud import *

from db import *

router = APIRouter()


@router.post("/login", response_model=Login)
async def login(user: User, db: Session = Depends(get_db)) -> Login:
    try:
        user = read_id_password(db, User.id, User.password)
        login = Login
        login.success = True
    except Exception as e:
        return HTTPException(
            status_code="500",
            detail="로그인 정보가 올바르지 않습니다",
        )
    return login
