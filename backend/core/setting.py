from pydantic import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):

    postgres_user: str
    postgres_password: str
    postgres_host: str
    postgres_port: str
    postgres_db_name: str
    stroage_path: str

    class Confing:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
