#!/usr/bin/env python3
"""Test script to verify Docker database setup."""

import os
import sys
import sqlite3
from pathlib import Path

def test_database_file():
    """Test if database file exists and is accessible."""
    print("🔍 Testing database file...")
    
    database_url = os.getenv('DATABASE_URL', 'sqlite:///app/data/arxiv_bot.db')
    
    if database_url.startswith('sqlite:///'):
        db_path = database_url.replace('sqlite:///', '')
        
        print(f"   Database path: {db_path}")
        
        if os.path.exists(db_path):
            print("   ✅ Database file exists")
            
            # Check file size
            size = os.path.getsize(db_path)
            print(f"   📊 Database size: {size} bytes")
            
            # Test SQLite connection
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # List tables
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = cursor.fetchall()
                
                print(f"   📋 Tables found: {len(tables)}")
                for table in tables:
                    print(f"      - {table[0]}")
                
                conn.close()
                print("   ✅ Database is accessible")
                return True
                
            except Exception as e:
                print(f"   ❌ Database connection failed: {e}")
                return False
        else:
            print("   ❌ Database file does not exist")
            return False
    else:
        print(f"   ℹ️  Non-SQLite database: {database_url}")
        return True

def test_arxiv_bot_imports():
    """Test if ArXiv bot modules can be imported."""
    print("\n🔍 Testing ArXiv bot imports...")
    
    try:
        from arxiv_bot.core.database import db_manager
        print("   ✅ Database manager imported")
        
        from arxiv_bot.core.models import Monitor, TrackedPaper, BotConfig
        print("   ✅ Database models imported")
        
        from arxiv_bot.core.config import settings
        print("   ✅ Configuration imported")
        
        return True
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        return False

def test_database_operations():
    """Test basic database operations."""
    print("\n🔍 Testing database operations...")
    
    try:
        from arxiv_bot.core.database import DatabaseManager
        from arxiv_bot.core.models import Monitor
        
        # Use corrected database URL
        db_manager = DatabaseManager("sqlite:////app/data/arxiv_bot.db")
        
        with db_manager.get_session() as session:
            # Count monitors
            count = session.query(Monitor).count()
            print(f"   📊 Monitor count: {count}")
            
            print("   ✅ Database operations working")
            return True
            
    except Exception as e:
        print(f"   ❌ Database operations failed: {e}")
        return False

def test_monitor_service():
    """Test monitor service functionality."""
    print("\n🔍 Testing monitor service...")
    
    try:
        from arxiv_bot.core.monitor_service import MonitorService
        
        service = MonitorService()
        print("   ✅ Monitor service created")
        
        # Test listing monitors (should work even if empty)
        monitors = service.list_monitors("test", "test_channel")
        print(f"   📊 Monitors for test channel: {len(monitors)}")
        
        print("   ✅ Monitor service working")
        return True
        
    except Exception as e:
        print(f"   ❌ Monitor service failed: {e}")
        return False

def main():
    """Run all Docker setup tests."""
    print("🐳 Docker ArXiv Bot Setup Test")
    print("=" * 40)
    
    # Set the corrected database URL early
    import os
    os.environ['DATABASE_URL'] = 'sqlite:////app/data/arxiv_bot.db'
    
    # Show environment info
    print(f"🐍 Python version: {sys.version}")
    print(f"📁 Working directory: {os.getcwd()}")
    print(f"🗃️  Database URL: {os.getenv('DATABASE_URL', 'sqlite:///app/data/arxiv_bot.db')}")
    print()
    
    tests = [
        test_database_file,
        test_arxiv_bot_imports,
        test_database_operations,
        test_monitor_service
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All Docker setup tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Check the logs above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 