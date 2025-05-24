"""Database management for ArXiv Bot."""

import sys
import os
from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from .config import settings
from .models import Base


def get_corrected_database_url():
    """Get the database URL with proper path correction for Docker environments."""
    database_url = settings.database_url
    
    # Fix SQLite path for Docker environments
    if database_url.startswith('sqlite:///'):
        db_path = database_url[10:]  # Remove 'sqlite:///'
        
        # If the path doesn't start with '/', it's relative and needs correction
        if not db_path.startswith('/'):
            db_path = '/' + db_path
        
        database_url = f"sqlite:///{db_path}"
    
    return database_url


class DatabaseManager:
    """Database connection and session management."""
    
    def __init__(self, database_url: str = None):
        self.database_url = database_url or get_corrected_database_url()
        self.engine = create_engine(self.database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
    
    def create_tables(self):
        """Create all tables in the database."""
        Base.metadata.create_all(bind=self.engine)
    
    def drop_tables(self):
        """Drop all tables in the database."""
        Base.metadata.drop_all(bind=self.engine)
    
    @contextmanager
    def get_session(self) -> Session:
        """Get a database session with automatic cleanup."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()


# Global database manager instance
db_manager = DatabaseManager()


def init_database():
    """Initialize the database by creating all tables."""
    print("Initializing database...")
    db_manager.create_tables()
    print("Database initialized successfully.")


def main():
    """CLI entry point for database operations."""
    if len(sys.argv) > 1 and sys.argv[1] == "init":
        init_database()
    else:
        print("Usage: python -m arxiv_bot.core.database init")


if __name__ == "__main__":
    main() 