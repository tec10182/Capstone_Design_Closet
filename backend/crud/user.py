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
        return HTTPException(status_code=500, detail="DB ERROR")

    return user


def read_id(db: Session, id: str):
    try:
        sql = text("SELECT * " "FROM users " "WHERE id = :id;")
        user = db.execute(sql, {"id": id}).first()
    except Exception as e:
        return HTTPException(status_code=500, detail="DB ERROR")

    return user


def read_password(db: Session, password: str):
    try:
        sql = text("SELECT * " "FROM users " "WHERE password = :password;")
        user = db.execute(sql, {"password": password}).first()
    except Exception as e:
        return HTTPException(status_code=500, detail="DB ERROR")

    return user


def write_user(db: Session, id: str, password: str):
    try:
        sql = text("INSERT INTO users (id, password) VALUES (:id, :password);")
        db.execute(sql, {"id": id, "password": password})
        db.commit()  # 변경 사항을 커밋합니다.
    except Exception as e:
        db.rollback()  # 오류 발생 시 롤백
        raise HTTPException(status_code=500, detail="DB ERROR")
    return {"success": True, "msg": "유저가 성공적으로 추가되었습니다"}
