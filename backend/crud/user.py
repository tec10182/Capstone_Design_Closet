from sqlalchemy.orm import Session
from sqlalchemy import and_, text
from sqlalchemy.sql.expression import func, distinct
from fastapi import HTTPException

# from utils import LOGGER
from model import *
from scheme import *


def read_id_password(db: Session, id: str, password: str):
    """id password가 존재하는 유저 정보를 읽음

    Args:
        db (Session): DB
        id (str): 아이디
        password (str): 비밀번호
    """
    try:
        sql = text("SELECT * " "FROM users " "WHERE id = :id AND password = :password;")
        user = db.execute(sql, {"id": id, "password": password}).first()
    except Exception as e:
        return HTTPException(status_code=500, detail="DB ERROR(please check category)")

    return user
