from sqlalchemy.orm import Session
from sqlalchemy import and_, text
from sqlalchemy.sql.expression import func, distinct
from fastapi import HTTPException


def read_image(db: Session, id: str):
    try:
        sql = text("SELECT img_name " "FROM items " "WHERE id = :id;")
        image = db.execute(sql, {"id": id}).first()
    except Exception as e:
        return HTTPException(status_code=500, detail="DB ERROR")

    return image
