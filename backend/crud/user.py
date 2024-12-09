from sqlalchemy.orm import Session
from sqlalchemy import and_, text
from sqlalchemy.sql.expression import func, distinct
from fastapi import HTTPException

from model import *
from scheme import *


def read_id_password(db: Session, id: str, password: str):
    """id password가 존재하는 유저 정보를 읽음"""
    try:
        sql = text("SELECT * " "FROM users " "WHERE id = :id AND password = :password;")
        user = db.execute(sql, {"id": id, "password": password}).first()
    except Exception as e:
        return HTTPException(status_code=500, detail="DB ERROR")
    return user


# read_image_from_id를 여기서만 직접 불러옵니다.
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

        return {"image": images, "category": category}
    except Exception as e:
        raise HTTPException(status_code=500, detail="DB ERROR")


# 나머지 함수들
def read_id(db: Session, id: str):
    try:
        sql = text("SELECT * " "FROM users " "WHERE id = :id;")
        user = db.execute(sql, {"id": id}).first()
    except Exception as e:
        return HTTPException(status_code=500, detail="DB ERROR")
    return user


def read_password(db: Session, password: str):
    try:
        sql = text("SELECT * FROM users WHERE password = :password;")
        user = db.execute(sql, {"password": password}).first()
    except Exception as e:
        raise HTTPException(status_code=500, detail="DB ERROR")
    return user


# backend/crud/user.py
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
    """주어진 id와 password로 사용자 정보를 삭제

    Args:
        db (Session): DB 세션
        id (str): 사용자 ID
        password (str): 사용자 비밀번호
    """
    try:
        # 유저 삭제 쿼리
        sql = text("DELETE FROM users WHERE id = :id AND password = :password;")
        result = db.execute(sql, {"id": id, "password": password})
        db.commit()  # 변경 사항을 커밋
    except Exception as e:
        db.rollback()  # 오류 발생 시 롤백
        raise HTTPException(status_code=500, detail="DB ERROR")

    return result


def update_category(db: Session, image_id: str, category: str):
    try:
        print(image_id, category)
        sql = text(
            "UPDATE items SET category = :category WHERE img_name = :image_name;"
        )
        result = db.execute(sql, {"image_name": image_id, "category": category})
        db.commit()

        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail="Item not found")
    except Exception as e:
        db.rollback()
        print
        raise HTTPException(status_code=500, detail="DB ERROR")

    return result


def delete_image(db: Session, image_name: str):
    try:
        sql = text("DELETE FROM items WHERE img_name = :image_name;")
        result = db.execute(sql, {"image_name": image_name})
        db.commit()

        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail="Item not found")
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail="DB ERROR")

    return result
