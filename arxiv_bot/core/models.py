"""Database models for ArXiv Bot."""

from datetime import datetime
from typing import Optional
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class Monitor(Base):
    """Model for paper monitoring configurations."""
    
    __tablename__ = "monitors"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    platform = Column(String(20), nullable=False)  # 'slack' or 'telegram'
    channel_id = Column(String(100), nullable=False)
    subject = Column(String(50), nullable=False)
    keywords = Column(String(500), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_checked = Column(DateTime, nullable=True)
    
    # Relationship with tracked papers
    tracked_papers = relationship("TrackedPaper", back_populates="monitor")
    
    def __repr__(self):
        return f"<Monitor(id={self.id}, platform={self.platform}, subject={self.subject})>"


class TrackedPaper(Base):
    """Model for tracking papers that have been processed."""
    
    __tablename__ = "tracked_papers"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    monitor_id = Column(Integer, ForeignKey("monitors.id"), nullable=False)
    arxiv_id = Column(String(50), nullable=False, unique=True)
    title = Column(String(500), nullable=False)
    authors = Column(Text, nullable=True)
    abstract = Column(Text, nullable=True)
    published_date = Column(DateTime, nullable=True)
    arxiv_url = Column(String(200), nullable=False)
    pdf_url = Column(String(200), nullable=True)
    processed_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship with monitor
    monitor = relationship("Monitor", back_populates="tracked_papers")
    
    def __repr__(self):
        return f"<TrackedPaper(id={self.id}, arxiv_id={self.arxiv_id}, title={self.title[:50]}...)>"


class BotConfig(Base):
    """Model for storing bot-specific configurations."""
    
    __tablename__ = "bot_configs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    platform = Column(String(20), nullable=False)
    channel_id = Column(String(100), nullable=False)
    config_key = Column(String(50), nullable=False)
    config_value = Column(String(200), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<BotConfig(platform={self.platform}, key={self.config_key}, value={self.config_value})>" 