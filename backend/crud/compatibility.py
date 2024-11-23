from sqlalchemy.orm import Session
from sqlalchemy import and_, text
from sqlalchemy.sql.expression import func, distinct
from fastapi import HTTPException


def read_image_embedding(db: Session, category: str, item_id: int):
    try:
        images = []
        embeddings = []

        # id와 category를 동시에 확인하는 SQL 쿼리
        sql = text(
            """
            SELECT img_name, embedding 
            FROM items 
            WHERE category = :category 
              AND id = :id;
        """
        )
        result = db.execute(sql, {"category": category, "id": item_id})

        # 결과에서 이미지 이름과 임베딩 값을 추출하여 리스트에 추가
        for row in result.fetchall():
            images.append(row[0])
            embeddings.append(row[1])

        # 결과가 없으면 빈 리스트로 반환
        if not images:
            raise HTTPException(
                status_code=404, detail="No images found for the given category and id"
            )

        return {"images": images, "embeddings": embeddings}

    except Exception as e:
        # 예외가 발생하면 서버 오류 반환
        raise HTTPException(status_code=500, detail="DB ERROR")


def read_embedding(db: Session, img_name: str):
    try:
        sql = text(
            """
            SELECT embedding 
            FROM items 
            WHERE img_name = :img_name
        """
        )
        embedding = db.execute(sql, {"img_name": img_name}).first()
    except Exception as e:
        return HTTPException(status_code=500, detail="DB ERROR")

    return embedding[0]
