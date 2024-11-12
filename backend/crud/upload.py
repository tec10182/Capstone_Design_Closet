from sqlalchemy.orm import Session
from sqlalchemy import and_, text
from sqlalchemy.sql.expression import func, distinct
from fastapi import HTTPException


def insert_image(
    db: Session,
    id: str,
    img_name: str,
    description: str,
    category: str,
    embedding: str,
):
    try:
        sql = text(
            "INSERT INTO items (id, img_name, description, category, embedding) VALUES (:id, :img_name, :description, :category, :embedding);"
        )
        db.execute(
            sql,
            {
                "id": id,
                "img_name": img_name,
                "description": description,
                "category": category,
                "embedding": embedding,
            },
        )
        db.commit()  # 변경 사항을 커밋합니다.
    except Exception as e:
        db.rollback()  # 오류 발생 시 롤백
        raise HTTPException(status_code=500, detail="DB ERROR")
    return
