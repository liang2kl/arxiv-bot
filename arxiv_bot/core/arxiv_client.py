"""ArXiv API client for fetching papers."""

import re
import arxiv
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class Paper:
    """Represents a paper from ArXiv."""
    
    arxiv_id: str
    title: str
    authors: List[str]
    abstract: str
    published_date: datetime
    arxiv_url: str
    pdf_url: str
    categories: List[str]
    
    @classmethod
    def from_arxiv_result(cls, result: arxiv.Result) -> 'Paper':
        """Create a Paper from an arxiv.Result object."""
        return cls(
            arxiv_id=result.entry_id.split('/')[-1],
            title=result.title.strip(),
            authors=[author.name for author in result.authors],
            abstract=result.summary.strip(),
            published_date=result.published,
            arxiv_url=result.entry_id,
            pdf_url=result.pdf_url,
            categories=[cat for cat in result.categories]
        )


class ArXivClient:
    """Client for interacting with ArXiv API."""
    
    def __init__(self):
        self.client = arxiv.Client()
    
    def search_papers(
        self,
        subject: str,
        keywords: str,
        max_results: int = 10,
        days_back: int = 7
    ) -> List[Paper]:
        """
        Search for papers by subject and keywords.
        
        Args:
            subject: ArXiv subject category (e.g., 'cs.AI', 'cs.LG')
            keywords: Keywords to search for in title and abstract
            max_results: Maximum number of results to return
            days_back: How many days back to search
            
        Returns:
            List of Paper objects
        """
        # Build search query
        query_parts = []
        
        # Add subject category
        if subject:
            query_parts.append(f"cat:{subject}")
        
        # Add keywords search in title and abstract
        if keywords:
            keyword_query = f"({keywords})"
            query_parts.append(f"(ti:{keyword_query} OR abs:{keyword_query})")
        
        # Combine query parts
        if not query_parts:
            raise ValueError("Must specify either subject or keywords")
        
        query = " AND ".join(query_parts)
        
        # Create search with date filter
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        
        # Fetch results and filter by date
        results = []
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        for result in self.client.results(search):
            # Make cutoff_date timezone-aware if result.published is timezone-aware
            result_published = result.published
            if result_published.tzinfo is not None and cutoff_date.tzinfo is None:
                from datetime import timezone
                cutoff_date = cutoff_date.replace(tzinfo=timezone.utc)
            elif result_published.tzinfo is None and cutoff_date.tzinfo is not None:
                cutoff_date = cutoff_date.replace(tzinfo=None)
            
            if result_published >= cutoff_date:
                results.append(Paper.from_arxiv_result(result))
        
        return results
    
    def get_paper_by_id(self, arxiv_id: str) -> Optional[Paper]:
        """
        Get a specific paper by ArXiv ID.
        
        Args:
            arxiv_id: The ArXiv ID (e.g., '2301.12345' or 'http://arxiv.org/abs/2301.12345')
            
        Returns:
            Paper object or None if not found
        """
        # Extract ID from URL if necessary
        if arxiv_id.startswith('http'):
            arxiv_id = arxiv_id.split('/')[-1]
        
        # Remove version number if present
        arxiv_id = re.sub(r'v\d+$', '', arxiv_id)
        
        search = arxiv.Search(id_list=[arxiv_id])
        
        try:
            result = next(self.client.results(search))
            return Paper.from_arxiv_result(result)
        except StopIteration:
            return None
    
    def get_paper_by_doi(self, doi: str) -> Optional[Paper]:
        """
        Get a paper by DOI (limited support).
        
        Args:
            doi: The DOI of the paper
            
        Returns:
            Paper object or None if not found
        """
        # ArXiv doesn't directly support DOI search, but we can try searching
        # by DOI in the abstract or comments
        search = arxiv.Search(
            query=f"abs:{doi}",
            max_results=1,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        try:
            result = next(self.client.results(search))
            return Paper.from_arxiv_result(result)
        except StopIteration:
            return None 