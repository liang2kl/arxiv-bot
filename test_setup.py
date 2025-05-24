#!/usr/bin/env python3
"""Test script to verify ArXiv Bot setup."""

import sys
import os

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import arxiv
        print("✅ arxiv package imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import arxiv: {e}")
        return False
    
    try:
        from arxiv_bot.core.config import settings
        print("✅ Core config imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import core config: {e}")
        return False
    
    try:
        from arxiv_bot.core.arxiv_client import ArXivClient
        print("✅ ArXiv client imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import ArXiv client: {e}")
        return False
    
    try:
        from arxiv_bot.core.database import db_manager
        print("✅ Database manager imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import database manager: {e}")
        return False
    
    return True

def test_arxiv_search():
    """Test basic ArXiv search functionality."""
    print("\nTesting ArXiv search...")
    
    try:
        from arxiv_bot.core.arxiv_client import ArXivClient
        
        client = ArXivClient()
        papers = client.search_papers(
            subject="cs.AI",
            keywords="transformer",
            max_results=2,
            days_back=30
        )
        
        print(f"✅ Found {len(papers)} papers")
        if papers:
            print(f"   Sample paper: {papers[0].title[:50]}...")
        
        return True
    except Exception as e:
        print(f"❌ ArXiv search failed: {e}")
        return False

def test_database():
    """Test database initialization."""
    print("\nTesting database...")
    
    try:
        from arxiv_bot.core.database import db_manager
        
        # Try to create tables
        db_manager.create_tables()
        print("✅ Database tables created successfully")
        
        # Test session
        with db_manager.get_session() as session:
            print("✅ Database session works")
        
        return True
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("ArXiv Bot Setup Test")
    print("=" * 20)
    
    tests = [
        test_imports,
        test_arxiv_search,
        test_database
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"\nResults: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! Your setup is ready.")
        print("\nNext steps:")
        print("1. Copy .env.example to .env and configure your API keys")
        print("2. Run: python -m arxiv_bot.core.database init")
        print("3. Start the bots:")
        print("   - Slack: python -m arxiv_bot.slack_bot")
        print("   - Telegram: python -m arxiv_bot.telegram_bot")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 