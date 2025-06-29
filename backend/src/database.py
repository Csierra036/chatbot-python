from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from src.config import settings

if settings.ENV == 'production':
    DATABASE_URL = f"postgresql://postgres.nqsrgdbrgwvhkijvdjps:{settings.DB_PASSWORD}@aws-0-us-east-2.pooler.supabase.com:5432/postgres"
else:
    DATABASE_URL = f"postgresql+psycopg2://{settings.DB_USERNAME}:{settings.DB_PASSWORD}@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"


engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()