"""AI providers for generating paper summaries."""

from openai import OpenAI
import google.generativeai as genai
import anthropic
from abc import ABC, abstractmethod
from typing import Optional
from .config import settings
from .arxiv_client import Paper


class AIProvider(ABC):
    """Abstract base class for AI providers."""
    
    @abstractmethod
    def generate_summary(self, paper: Paper) -> str:
        """Generate a summary for the given paper."""
        pass


class OpenAIProvider(AIProvider):
    """OpenAI GPT provider for generating summaries."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or settings.openai_api_key
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = OpenAI(api_key=self.api_key)
    
    def generate_summary(self, paper: Paper) -> str:
        """Generate a summary using OpenAI GPT."""
        prompt = self._build_prompt(paper)
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a research assistant who specializes in creating clear, comprehensive summaries of academic papers. Focus on the key contributions, methodology, and implications."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            return f"Error generating summary with OpenAI: {str(e)}"
    
    def _build_prompt(self, paper: Paper) -> str:
        """Build the prompt for OpenAI."""
        authors_str = ", ".join(paper.authors[:3])  # Limit to first 3 authors
        if len(paper.authors) > 3:
            authors_str += " et al."
        
        return f"""
Please provide a comprehensive summary of the following research paper:

Title: {paper.title}
Authors: {authors_str}
Categories: {', '.join(paper.categories)}

Abstract:
{paper.abstract}

Please include:
1. Main research question and objectives
2. Key methodology and approach
3. Major findings and contributions
4. Potential implications and applications
5. Limitations or future work mentioned

Keep the summary clear and accessible while maintaining technical accuracy.
"""


class GoogleProvider(AIProvider):
    """Google Gemini provider for generating summaries."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or settings.google_api_key
        if not self.api_key:
            raise ValueError("Google API key is required")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-pro')
    
    def generate_summary(self, paper: Paper) -> str:
        """Generate a summary using Google Gemini."""
        prompt = self._build_prompt(paper)
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=500,
                    temperature=0.3
                )
            )
            
            return response.text.strip()
        
        except Exception as e:
            return f"Error generating summary with Google Gemini: {str(e)}"
    
    def _build_prompt(self, paper: Paper) -> str:
        """Build the prompt for Google Gemini."""
        authors_str = ", ".join(paper.authors[:3])
        if len(paper.authors) > 3:
            authors_str += " et al."
        
        return f"""
Analyze and summarize this research paper comprehensively:

**Title:** {paper.title}
**Authors:** {authors_str}
**Categories:** {', '.join(paper.categories)}

**Abstract:**
{paper.abstract}

Please provide a structured summary covering:
1. Research objectives and motivation
2. Methodology and technical approach  
3. Key findings and results
4. Significance and potential impact
5. Future research directions

Make it informative yet accessible to researchers in related fields.
"""


class AnthropicProvider(AIProvider):
    """Anthropic Claude provider for generating summaries."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or settings.anthropic_api_key
        if not self.api_key:
            raise ValueError("Anthropic API key is required")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
    
    def generate_summary(self, paper: Paper) -> str:
        """Generate a summary using Anthropic Claude."""
        prompt = self._build_prompt(paper)
        
        try:
            response = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=500,
                temperature=0.3,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            return response.content[0].text.strip()
        
        except Exception as e:
            return f"Error generating summary with Anthropic Claude: {str(e)}"
    
    def _build_prompt(self, paper: Paper) -> str:
        """Build the prompt for Anthropic Claude."""
        authors_str = ", ".join(paper.authors[:3])
        if len(paper.authors) > 3:
            authors_str += " et al."
        
        return f"""
Please create a detailed summary of this academic paper:

Title: {paper.title}
Authors: {authors_str}
Subject Areas: {', '.join(paper.categories)}

Abstract:
{paper.abstract}

Provide a comprehensive analysis including:
• Research problem and motivation
• Methodology and experimental design
• Key contributions and findings
• Practical implications
• Strengths and potential limitations

Aim for clarity while preserving technical depth appropriate for academic audiences.
"""


class AIProviderFactory:
    """Factory for creating AI provider instances."""
    
    _providers = {
        'openai': OpenAIProvider,
        'google': GoogleProvider,
        'anthropic': AnthropicProvider
    }
    
    @classmethod
    def create_provider(cls, provider_name: str) -> AIProvider:
        """Create an AI provider instance."""
        provider_name = provider_name.lower()
        
        if provider_name not in cls._providers:
            available = ', '.join(cls._providers.keys())
            raise ValueError(f"Unknown provider '{provider_name}'. Available: {available}")
        
        provider_class = cls._providers[provider_name]
        return provider_class()
    
    @classmethod
    def get_default_provider(cls) -> AIProvider:
        """Get the default AI provider."""
        return cls.create_provider(settings.default_ai_provider)
    
    @classmethod
    def list_providers(cls) -> list:
        """List all available providers."""
        return list(cls._providers.keys()) 