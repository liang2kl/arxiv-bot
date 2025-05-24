#!/usr/bin/env python3
"""
Robust database initialization script for ArXiv Bot.
This script ensures the database is properly created and initialized.
"""

import os
import sys
import logging
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def setup_logging():
    """Set up logging for the initialization process."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def check_database_path():
    """Check and create the database directory if needed."""
    logger = logging.getLogger(__name__)
    
    # Get database URL from environment or use default
    database_url = os.getenv('DATABASE_URL', 'sqlite:///app/data/arxiv_bot.db')
    
    if database_url.startswith('sqlite:///'):
        # Remove the sqlite:/// prefix
        db_path = database_url[10:]  # Remove 'sqlite:///'
        
        # If the path starts with '/', it's already absolute
        if not db_path.startswith('/'):
            # For relative paths, make them absolute from root
            db_path = '/' + db_path
        
        db_dir = os.path.dirname(db_path)
        
        logger.info(f"Database URL: {database_url}")
        logger.info(f"Database path: {db_path}")
        logger.info(f"Database directory: {db_dir}")
        
        # Create directory if it doesn't exist
        if db_dir and not os.path.exists(db_dir):
            logger.info(f"Creating database directory: {db_dir}")
            os.makedirs(db_dir, exist_ok=True)
        
        # Check if database already exists
        if os.path.exists(db_path):
            logger.info(f"Database already exists at: {db_path}")
            return db_path, True
        else:
            logger.info(f"Database will be created at: {db_path}")
            return db_path, False
    
    return None, False

def initialize_database():
    """Initialize the database with proper error handling."""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting database initialization...")
        
        # Check database path
        db_path, exists = check_database_path()
        
        # Import database components
        logger.info("Importing database components...")
        from arxiv_bot.core.database import DatabaseManager
        from arxiv_bot.core.models import Base
        
        # Create database manager with corrected URL
        corrected_url = f"sqlite:///{db_path}"
        logger.info(f"Using corrected database URL: {corrected_url}")
        db_manager = DatabaseManager(corrected_url)
        
        # Create tables
        logger.info("Creating database tables...")
        db_manager.create_tables()
        
        # Test database connection
        logger.info("Testing database connection...")
        with db_manager.get_session() as session:
            # Simple test query
            from arxiv_bot.core.models import Monitor
            count = session.query(Monitor).count()
            logger.info(f"Database connection successful. Monitor count: {count}")
        
        # Verify database file was created (for SQLite)
        if db_path and os.path.exists(db_path):
            file_size = os.path.getsize(db_path)
            logger.info(f"Database file created successfully: {db_path} ({file_size} bytes)")
        
        logger.info("Database initialization completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main initialization function."""
    logger = setup_logging()
    
    logger.info("ArXiv Bot Database Initialization")
    logger.info("=" * 40)
    
    # Show environment info
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Database URL: {os.getenv('DATABASE_URL', 'sqlite:///app/data/arxiv_bot.db')}")
    
    # Initialize database
    success = initialize_database()
    
    if success:
        logger.info("✅ Database initialization successful!")
        sys.exit(0)
    else:
        logger.error("❌ Database initialization failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 