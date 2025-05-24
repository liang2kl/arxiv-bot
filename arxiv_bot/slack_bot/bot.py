"""Slack bot implementation for ArXiv monitoring and summarization."""

import logging
import threading
import time
from typing import List
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from ..core.config import settings
from ..core.monitor_service import MonitorService
from ..core.arxiv_client import Paper
from ..core.ai_providers import AIProviderFactory

logger = logging.getLogger(__name__)


class SlackBot:
    """Slack bot for ArXiv paper monitoring and AI summarization."""
    
    def __init__(self):
        self.app = App(token=settings.slack_bot_token)
        self.monitor_service = MonitorService()
        self.ai_factory = AIProviderFactory()
        self._setup_commands()
        self._setup_monitoring_thread()
    
    def _setup_commands(self):
        """Set up Slack slash commands."""
        
        @self.app.command("/arxiv_monitor")
        def handle_monitor_command(ack, respond, command):
            ack()
            try:
                # Parse command arguments
                args = command['text'].strip().split()
                if len(args) < 2:
                    respond("Usage: `/arxiv_monitor <subject> <keywords>`\n"
                           "Example: `/arxiv_monitor cs.AI transformer attention`")
                    return
                
                subject = args[0]
                keywords = ' '.join(args[1:])
                channel_id = command['channel_id']
                
                # Create monitor
                monitor_id = self.monitor_service.create_monitor(
                    platform='slack',
                    channel_id=channel_id,
                    subject=subject,
                    keywords=keywords
                )
                
                respond(f"✅ Created monitor #{monitor_id} for subject `{subject}` with keywords: `{keywords}`\n"
                       f"I'll check for new papers every {settings.monitor_interval_hours} hours.")
                
            except Exception as e:
                logger.error(f"Error in monitor command: {str(e)}")
                respond(f"❌ Error creating monitor: {str(e)}")
        
        @self.app.command("/arxiv_stop")
        def handle_stop_command(ack, respond, command):
            ack()
            try:
                args = command['text'].strip().split()
                if len(args) != 1:
                    respond("Usage: `/arxiv_stop <monitor_id>`")
                    return
                
                monitor_id = int(args[0])
                success = self.monitor_service.stop_monitor(monitor_id)
                
                if success:
                    respond(f"✅ Stopped monitor #{monitor_id}")
                else:
                    respond(f"❌ Monitor #{monitor_id} not found")
                    
            except ValueError:
                respond("❌ Invalid monitor ID. Please provide a number.")
            except Exception as e:
                logger.error(f"Error in stop command: {str(e)}")
                respond(f"❌ Error stopping monitor: {str(e)}")
        
        @self.app.command("/arxiv_list")
        def handle_list_command(ack, respond, command):
            ack()
            try:
                channel_id = command['channel_id']
                monitors = self.monitor_service.list_monitors('slack', channel_id)
                
                if not monitors:
                    respond("No active monitors in this channel.")
                    return
                
                response = "📋 *Active Monitors:*\n\n"
                for monitor in monitors:
                    response += f"*#{monitor['id']}* - `{monitor['subject']}` | Keywords: `{monitor['keywords']}`\n"
                    response += f"  Created: {monitor['created_at'].strftime('%Y-%m-%d %H:%M')}"
                    if monitor['last_checked']:
                        response += f" | Last checked: {monitor['last_checked'].strftime('%Y-%m-%d %H:%M')}"
                    response += f" | Papers found: {monitor['paper_count']}\n\n"
                
                respond(response)
                
            except Exception as e:
                logger.error(f"Error in list command: {str(e)}")
                respond(f"❌ Error listing monitors: {str(e)}")
        
        @self.app.command("/arxiv_summarize")
        def handle_summarize_command(ack, respond, command):
            ack()
            try:
                paper_id = command['text'].strip()
                if not paper_id:
                    respond("Usage: `/arxiv_summarize <arxiv_url_or_id_or_doi>`\n"
                           "Example: `/arxiv_summarize https://arxiv.org/abs/2301.12345`")
                    return
                
                respond("🔄 Generating summary... This may take a moment.")
                
                # Get configured AI provider for this channel
                channel_id = command['channel_id']
                provider_name = self.monitor_service.get_bot_config(
                    'slack', channel_id, 'ai_provider'
                )
                
                paper, summary = self.monitor_service.generate_summary(
                    paper_id, provider_name
                )
                
                response = self._format_paper_summary(paper, summary)
                respond(response)
                
            except Exception as e:
                logger.error(f"Error in summarize command: {str(e)}")
                respond(f"❌ Error generating summary: {str(e)}")
        
        @self.app.command("/arxiv_config")
        def handle_config_command(ack, respond, command):
            ack()
            try:
                args = command['text'].strip().split()
                if len(args) < 2:
                    providers = self.ai_factory.list_providers()
                    respond(f"Usage: `/arxiv_config provider <provider_name>`\n"
                           f"Available providers: {', '.join(providers)}")
                    return
                
                if args[0] == 'provider':
                    provider_name = args[1].lower()
                    available_providers = self.ai_factory.list_providers()
                    
                    if provider_name not in available_providers:
                        respond(f"❌ Unknown provider. Available: {', '.join(available_providers)}")
                        return
                    
                    channel_id = command['channel_id']
                    self.monitor_service.set_bot_config(
                        'slack', channel_id, 'ai_provider', provider_name
                    )
                    
                    respond(f"✅ Set AI provider to `{provider_name}` for this channel")
                else:
                    respond("❌ Unknown config option. Use `provider` to set AI provider.")
                    
            except Exception as e:
                logger.error(f"Error in config command: {str(e)}")
                respond(f"❌ Error updating config: {str(e)}")
    
    def _setup_monitoring_thread(self):
        """Set up background thread for monitoring papers."""
        def monitor_loop():
            while True:
                try:
                    logger.info("Checking for new papers...")
                    self.monitor_service.check_all_monitors(self._handle_new_papers)
                    time.sleep(settings.monitor_interval_hours * 3600)  # Convert hours to seconds
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {str(e)}")
                    time.sleep(300)  # Wait 5 minutes before retrying
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        logger.info(f"Started monitoring thread (interval: {settings.monitor_interval_hours} hours)")
    
    def _handle_new_papers(self, platform: str, channel_id: str, papers: List[Paper]):
        """Handle new papers found by monitors."""
        if platform != 'slack':
            return
        
        for paper in papers[:settings.max_papers_per_check]:
            try:
                message = self._format_paper_notification(paper)
                self.app.client.chat_postMessage(
                    channel=channel_id,
                    text=message,
                    unfurl_links=False
                )
            except Exception as e:
                logger.error(f"Error posting paper to Slack: {str(e)}")
    
    def _format_paper_notification(self, paper: Paper) -> str:
        """Format a paper for Slack notification."""
        authors = ", ".join(paper.authors[:3])
        if len(paper.authors) > 3:
            authors += " et al."
        
        message = f"📄 *New Paper Alert*\n\n"
        message += f"*{paper.title}*\n"
        message += f"👥 {authors}\n"
        message += f"📂 {', '.join(paper.categories)}\n"
        message += f"📅 Published: {paper.published_date.strftime('%Y-%m-%d')}\n\n"
        message += f"📖 *Abstract:*\n{paper.abstract[:300]}{'...' if len(paper.abstract) > 300 else ''}\n\n"
        message += f"🔗 <{paper.arxiv_url}|View on ArXiv> | <{paper.pdf_url}|Download PDF>"
        
        return message
    
    def _format_paper_summary(self, paper: Paper, summary: str) -> str:
        """Format a paper summary for Slack."""
        authors = ", ".join(paper.authors[:3])
        if len(paper.authors) > 3:
            authors += " et al."
        
        message = f"📄 *Paper Summary*\n\n"
        message += f"*{paper.title}*\n"
        message += f"👥 {authors}\n"
        message += f"📂 {', '.join(paper.categories)}\n"
        message += f"📅 Published: {paper.published_date.strftime('%Y-%m-%d')}\n\n"
        message += f"🤖 *AI Summary:*\n{summary}\n\n"
        message += f"🔗 <{paper.arxiv_url}|View on ArXiv> | <{paper.pdf_url}|Download PDF>"
        
        return message
    
    def start(self):
        """Start the Slack bot."""
        if not settings.slack_app_token:
            logger.error("SLACK_APP_TOKEN is required for socket mode")
            return
        
        handler = SocketModeHandler(self.app, settings.slack_app_token)
        logger.info("Starting Slack bot...")
        handler.start()


def main():
    """Main entry point for the Slack bot."""
    logging.basicConfig(level=logging.INFO)
    bot = SlackBot()
    bot.start()


if __name__ == "__main__":
    main() 