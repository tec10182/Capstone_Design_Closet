import psycopg2
from psycopg2 import sql


# PostgreSQL 연결 함수
def connect_to_postgres(dbname=None, autocommit=False):
    connection = psycopg2.connect(
        dbname=dbname if dbname else "postgres",  # 처음에는 dbname 없이 연결
        user="postgres",
        password="1234",
        host="localhost",
        port="5432",
    )

    if autocommit:
        connection.autocommit = True  # autocommit을 활성화
    return connection


# 데이터베이스 생성 함수
def create_database():
    connection = connect_to_postgres(autocommit=True)  # autocommit 활성화된 연결
    cursor = connection.cursor()

    try:
        cursor.execute("CREATE DATABASE closet;")
        print("Database 'closet' created successfully")
    except psycopg2.errors.DuplicateDatabase:
        print("Database 'closet' already exists")
    finally:
        cursor.close()
        connection.close()


# 테이블 생성 함수
def create_table():
    connection = connect_to_postgres(dbname="closet")
    cursor = connection.cursor()

    create_table_query = """
    CREATE TABLE IF NOT EXISTS users (
        id VARCHAR(255) UNIQUE NOT NULL,
        password VARCHAR(255) UNIQUE NOT NULL
    );
    """

    try:
        cursor.execute(create_table_query)
        connection.commit()
        print("Table 'users' created successfully")
    except Exception as e:
        print(f"Error creating table: {e}")
    finally:
        cursor.close()
        connection.close()


# 데이터 삽입 함수
def insert_user(user_id, password):
    connection = connect_to_postgres(dbname="closet")
    cursor = connection.cursor()

    insert_query = """
    INSERT INTO users (id, password) 
    VALUES (%s, %s)
    ON CONFLICT (id) DO NOTHING;
    """

    try:
        cursor.execute(insert_query, (user_id, password))
        connection.commit()
        print(f"User {user_id} inserted successfully")
    except Exception as e:
        print(f"Error inserting user {user_id}: {e}")
        connection.rollback()
    finally:
        cursor.close()
        connection.close()


# 사용자 삭제 함수
def delete_user(user_id):
    connection = connect_to_postgres(dbname="closet")
    cursor = connection.cursor()

    delete_query = """
    DELETE FROM users 
    WHERE id = %s;
    """

    try:
        cursor.execute(delete_query, (user_id,))
        connection.commit()
        if cursor.rowcount > 0:
            print(f"User {user_id} deleted successfully")
        else:
            print(f"User {user_id} not found")
    except Exception as e:
        print(f"Error deleting user {user_id}: {e}")
        connection.rollback()
    finally:
        cursor.close()
        connection.close()


# 새로운 테이블 생성 함수
def create_items_table():
    connection = connect_to_postgres(dbname="closet")
    cursor = connection.cursor()

    create_items_table_query = """
    CREATE TABLE IF NOT EXISTS items (
        id VARCHAR(255),
        img_name VARCHAR(255) UNIQUE,
        description VARCHAR(255),
        category VARCHAR(255),
        embedding VARCHAR(255)
    );
    """

    try:
        cursor.execute(create_items_table_query)
        connection.commit()
        print("Table 'items' created successfully")
    except Exception as e:
        print(f"Error creating table 'items': {e}")
    finally:
        cursor.close()
        connection.close()


# 아이템 삽입 함수
def insert_item(id, img_name, description, category, embedding):
    connection = connect_to_postgres(dbname="closet")
    cursor = connection.cursor()

    insert_item_query = """
    INSERT INTO items (id, img_name, description, category, embedding)
    VALUES (%s, %s, %s, %s, %s)
    ON CONFLICT (img_name) DO NOTHING;
    """

    try:
        cursor.execute(
            insert_item_query, (id, img_name, description, category, embedding)
        )
        connection.commit()
        print(f"Item '{img_name}' inserted successfully")
    except Exception as e:
        print(f"Error inserting item '{img_name}': {e}")
        connection.rollback()
    finally:
        cursor.close()
        connection.close()


def clear_tables():
    connection = connect_to_postgres(dbname="closet")
    cursor = connection.cursor()

    try:
        # 모든 테이블 데이터 삭제
        cursor.execute("TRUNCATE TABLE users, items RESTART IDENTITY CASCADE;")
        connection.commit()
        print("All tables cleared successfully")
    except Exception as e:
        print(f"Error clearing tables: {e}")
        connection.rollback()
    finally:
        cursor.close()
        connection.close()


def db_setting():
    create_database()  # 데이터베이스 생성
    create_table()  # users 테이블 생성
    create_items_table()  # items 테이블 생성
    clear_tables()
    insert_user("user1", "password123")  # users 테이블에 데이터 삽입
    insert_user("user2", "password456")  # users 테이블에 다른 유저 삽입


if __name__ == "__main__":
    create_database()  # 데이터베이스 생성
    create_table()  # users 테이블 생성
    create_items_table()  # items 테이블 생성
    clear_tables()
    insert_user("user1", "password123")  # users 테이블에 데이터 삽입
    insert_user("user2", "password456")  # users 테이블에 다른 유저 삽입
    # delete_user("user1")  # users 테이블에서 user1 삭제
    # insert_item(
    #     "user1", "1.png", "test", "top", "test.npy"
    # )  # items 테이블에 데이터 삽입
