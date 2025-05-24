"""Paper monitoring service for periodic ArXiv searches."""

import logging
from datetime import datetime, timedelta
from typing import List, Optional, Callable, Dict, Any
from sqlalchemy.orm import Session
from .database import db_manager
from .models import Monitor, TrackedPaper, BotConfig
from .arxiv_client import ArXivClient, Paper
from .ai_providers import AIProviderFactory

logger = logging.getLogger(__name__)


class MonitorService:
    """Service for managing paper monitoring and periodic checks."""
    
    def __init__(self):
        self.arxiv_client = ArXivClient()
        self.ai_factory = AIProviderFactory()
    
    def create_monitor(
        self,
        platform: str,
        channel_id: str,
        subject: str,
        keywords: str
    ) -> int:
        """
        Create a new monitor configuration.
        
        Args:
            platform: Platform name ('slack' or 'telegram')
            channel_id: Channel/chat ID
            subject: ArXiv subject category
            keywords: Keywords to search for
            
        Returns:
            Monitor ID
        """
        with db_manager.get_session() as session:
            monitor = Monitor(
                platform=platform,
                channel_id=channel_id,
                subject=subject,
                keywords=keywords,
                is_active=True
            )
            session.add(monitor)
            session.flush()
            monitor_id = monitor.id
            logger.info(f"Created monitor {monitor_id} for {platform} channel {channel_id}")
            return monitor_id
    
    def stop_monitor(self, monitor_id: int) -> bool:
        """
        Stop a monitor by setting it as inactive.
        
        Args:
            monitor_id: ID of the monitor to stop
            
        Returns:
            True if monitor was found and stopped, False otherwise
        """
        with db_manager.get_session() as session:
            monitor = session.query(Monitor).filter(Monitor.id == monitor_id).first()
            if monitor:
                monitor.is_active = False
                monitor.updated_at = datetime.utcnow()
                logger.info(f"Stopped monitor {monitor_id}")
                return True
            return False
    
    def list_monitors(self, platform: str, channel_id: str) -> List[Dict[str, Any]]:
        """
        List all active monitors for a specific platform and channel.
        
        Args:
            platform: Platform name
            channel_id: Channel/chat ID
            
        Returns:
            List of monitor information
        """
        with db_manager.get_session() as session:
            monitors = session.query(Monitor).filter(
                Monitor.platform == platform,
                Monitor.channel_id == channel_id,
                Monitor.is_active == True
            ).all()
            
            result = []
            for monitor in monitors:
                result.append({
                    'id': monitor.id,
                    'subject': monitor.subject,
                    'keywords': monitor.keywords,
                    'created_at': monitor.created_at,
                    'last_checked': monitor.last_checked,
                    'paper_count': len(monitor.tracked_papers)
                })
            
            return result
    
    def check_all_monitors(self, callback: Callable[[str, str, List[Paper]], None]):
        """
        Check all active monitors for new papers.
        
        Args:
            callback: Function to call when new papers are found.
                     Should accept (platform, channel_id, papers)
        """
        with db_manager.get_session() as session:
            active_monitors = session.query(Monitor).filter(
                Monitor.is_active == True
            ).all()
            
            for monitor in active_monitors:
                try:
                    new_papers = self._check_monitor(session, monitor)
                    if new_papers:
                        callback(monitor.platform, monitor.channel_id, new_papers)
                        logger.info(f"Found {len(new_papers)} new papers for monitor {monitor.id}")
                    
                    # Update last checked time
                    monitor.last_checked = datetime.utcnow()
                    session.commit()
                    
                except Exception as e:
                    logger.error(f"Error checking monitor {monitor.id}: {str(e)}")
                    session.rollback()
    
    def _check_monitor(self, session: Session, monitor: Monitor) -> List[Paper]:
        """
        Check a specific monitor for new papers.
        
        Args:
            session: Database session
            monitor: Monitor configuration
            
        Returns:
            List of new papers found
        """
        # Determine how far back to search
        days_back = 7  # Default
        if monitor.last_checked:
            days_since_check = (datetime.utcnow() - monitor.last_checked).days
            days_back = max(1, min(days_since_check + 1, 30))  # Cap at 30 days
        
        # Search for papers
        papers = self.arxiv_client.search_papers(
            subject=monitor.subject,
            keywords=monitor.keywords,
            max_results=50,  # Check more papers to avoid missing any
            days_back=days_back
        )
        
        # Filter out papers we've already tracked
        tracked_arxiv_ids = set()
        tracked_papers = session.query(TrackedPaper).filter(
            TrackedPaper.monitor_id == monitor.id
        ).all()
        
        for tracked in tracked_papers:
            tracked_arxiv_ids.add(tracked.arxiv_id)
        
        new_papers = []
        for paper in papers:
            if paper.arxiv_id not in tracked_arxiv_ids:
                # Add to tracked papers
                tracked_paper = TrackedPaper(
                    monitor_id=monitor.id,
                    arxiv_id=paper.arxiv_id,
                    title=paper.title,
                    authors=", ".join(paper.authors),
                    abstract=paper.abstract,
                    published_date=paper.published_date,
                    arxiv_url=paper.arxiv_url,
                    pdf_url=paper.pdf_url
                )
                session.add(tracked_paper)
                new_papers.append(paper)
        
        return new_papers
    
    def generate_summary(self, paper_identifier: str, provider_name: str = None) -> tuple[Paper, str]:
        """
        Generate an AI summary for a paper.
        
        Args:
            paper_identifier: ArXiv ID, URL, or DOI
            provider_name: AI provider to use (optional, uses default if not specified)
            
        Returns:
            Tuple of (Paper, summary)
        """
        # Get the paper
        paper = None
        
        if paper_identifier.startswith('http'):
            # ArXiv URL
            paper = self.arxiv_client.get_paper_by_id(paper_identifier)
        elif '/' in paper_identifier or paper_identifier.startswith('10.'):
            # Likely a DOI
            paper = self.arxiv_client.get_paper_by_doi(paper_identifier)
        else:
            # Assume ArXiv ID
            paper = self.arxiv_client.get_paper_by_id(paper_identifier)
        
        if not paper:
            raise ValueError(f"Could not find paper: {paper_identifier}")
        
        # Generate summary
        if provider_name:
            ai_provider = self.ai_factory.create_provider(provider_name)
        else:
            ai_provider = self.ai_factory.get_default_provider()
        
        summary = ai_provider.generate_summary(paper)
        return paper, summary
    
    def set_bot_config(self, platform: str, channel_id: str, key: str, value: str):
        """
        Set a bot configuration value.
        
        Args:
            platform: Platform name
            channel_id: Channel/chat ID
            key: Configuration key
            value: Configuration value
        """
        with db_manager.get_session() as session:
            # Check if config already exists
            config = session.query(BotConfig).filter(
                BotConfig.platform == platform,
                BotConfig.channel_id == channel_id,
                BotConfig.config_key == key
            ).first()
            
            if config:
                config.config_value = value
                config.updated_at = datetime.utcnow()
            else:
                config = BotConfig(
                    platform=platform,
                    channel_id=channel_id,
                    config_key=key,
                    config_value=value
                )
                session.add(config)
            
            logger.info(f"Set {platform} config {key}={value} for channel {channel_id}")
    
    def get_bot_config(self, platform: str, channel_id: str, key: str, default: str = None) -> Optional[str]:
        """
        Get a bot configuration value.
        
        Args:
            platform: Platform name
            channel_id: Channel/chat ID
            key: Configuration key
            default: Default value if not found
            
        Returns:
            Configuration value or default
        """
        with db_manager.get_session() as session:
            config = session.query(BotConfig).filter(
                BotConfig.platform == platform,
                BotConfig.channel_id == channel_id,
                BotConfig.config_key == key
            ).first()
            
            return config.config_value if config else default 