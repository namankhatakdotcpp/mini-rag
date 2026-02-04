from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.core.config import settings


def _normalize_db_url(url: str) -> str:
    """
    Ensure SQLAlchemy uses the psycopg (v3) driver.
    """
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


engine = create_engine(_normalize_db_url(settings.supabase_db_url), pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)


def get_db():
    """
    Dependency-injected database session.
    Ensures proper cleanup after request.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
