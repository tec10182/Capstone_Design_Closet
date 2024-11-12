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
    return


def erase_user(db: Session, id: str, password: str):
    try:
        sql = text("DELETE FROM users WHERE id = :id AND password = :password;")
        result = db.execute(sql, {"id": id, "password": password})
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail="DB ERROR")

    return


def read_image_from_id(db: Session, id: str):
    try:
        images = []
        category = []
        sql = text(
            """
            SELECT img_name, category 
            FROM items 
            WHERE id = :id; 
            """
        )
        result = db.execute(sql, {"id": id})

        for row in result.fetchall():
            images.append(row[0])
            category.append(row[1])
        print(1)

        return {"image": images, "category": category}
    except Exception as e:
        # 예외가 발생하면 서버 오류 반환
        raise HTTPException(status_code=500, detail="DB ERROR")
