#!/usr/bin/env python3
"""Test script to debug database path issues."""

import os
from sqlalchemy import create_engine, text

print("Working directory:", os.getcwd())
print("Database URL: sqlite:///app/data/arxiv_bot.db")

# Test the database creation
engine = create_engine('sqlite:///app/data/arxiv_bot.db')
print("Engine URL:", engine.url)

print("Files in /app/data before:", os.listdir('/app/data') if os.path.exists('/app/data') else 'Directory does not exist')

# Create a simple table to trigger database file creation
with engine.connect() as conn:
    conn.execute(text('CREATE TABLE IF NOT EXISTS test (id INTEGER)'))
    conn.commit()

print("Files in /app/data after:", os.listdir('/app/data') if os.path.exists('/app/data') else 'Directory does not exist')
print("Files in current dir after:", [f for f in os.listdir('.') if f.endswith('.db')])

# Check if the database file exists at the expected location
expected_path = '/app/data/arxiv_bot.db'
if os.path.exists(expected_path):
    print(f"✅ Database file found at: {expected_path}")
    print(f"   File size: {os.path.getsize(expected_path)} bytes")
else:
    print(f"❌ Database file NOT found at: {expected_path}")
    
    # Look for the database file elsewhere
    for root, dirs, files in os.walk('/app'):
        for file in files:
            if file.endswith('.db'):
                full_path = os.path.join(root, file)
                print(f"   Found database file at: {full_path}") 