"""Telegram bot implementation for ArXiv monitoring and summarization."""

import logging
import threading
import time
from typing import List
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, ContextTypes
from telegram.constants import ParseMode
from ..core.config import settings
from ..core.monitor_service import MonitorService
from ..core.arxiv_client import Paper
from ..core.ai_providers import AIProviderFactory

logger = logging.getLogger(__name__)


class TelegramBot:
    """Telegram bot for ArXiv paper monitoring and AI summarization."""
    
    def __init__(self):
        self.monitor_service = MonitorService()
        self.ai_factory = AIProviderFactory()
        self.application = Application.builder().token(settings.telegram_bot_token).build()
        self._setup_commands()
        self._setup_monitoring_thread()
    
    def _setup_commands(self):
        """Set up Telegram bot commands."""
        
        async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
            """Handle /start command."""
            welcome_message = """
🤖 *ArXiv Bot* - Your research paper assistant!

*Available Commands:*
• `/monitor <subject> <keywords>` - Start monitoring papers
• `/stop <monitor_id>` - Stop a specific monitor
• `/list` - List your active monitors
• `/summarize <arxiv_url_or_id>` - Get AI summary of a paper
• `/config provider <provider>` - Set AI provider (openai, google, anthropic)
• `/help` - Show this help message

*Example:*
`/monitor cs.AI transformer attention`

Let me know if you need help getting started! 📚
            """
            await update.message.reply_text(welcome_message, parse_mode=ParseMode.MARKDOWN)
        
        async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
            """Handle /help command."""
            await start_command(update, context)
        
        async def monitor_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
            """Handle /monitor command."""
            try:
                if len(context.args) < 2:
                    await update.message.reply_text(
                        "Usage: `/monitor <subject> <keywords>`\n"
                        "Example: `/monitor cs.AI transformer attention`",
                        parse_mode=ParseMode.MARKDOWN
                    )
                    return
                
                subject = context.args[0]
                keywords = ' '.join(context.args[1:])
                chat_id = str(update.effective_chat.id)
                
                # Create monitor
                monitor_id = self.monitor_service.create_monitor(
                    platform='telegram',
                    channel_id=chat_id,
                    subject=subject,
                    keywords=keywords
                )
                
                await update.message.reply_text(
                    f"✅ Created monitor #{monitor_id} for subject `{subject}` with keywords: `{keywords}`\n"
                    f"I'll check for new papers every {settings.monitor_interval_hours} hours.",
                    parse_mode=ParseMode.MARKDOWN
                )
                
            except Exception as e:
                logger.error(f"Error in monitor command: {str(e)}")
                await update.message.reply_text(f"❌ Error creating monitor: {str(e)}")
        
        async def stop_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
            """Handle /stop command."""
            try:
                if len(context.args) != 1:
                    await update.message.reply_text("Usage: `/stop <monitor_id>`", parse_mode=ParseMode.MARKDOWN)
                    return
                
                monitor_id = int(context.args[0])
                success = self.monitor_service.stop_monitor(monitor_id)
                
                if success:
                    await update.message.reply_text(f"✅ Stopped monitor #{monitor_id}")
                else:
                    await update.message.reply_text(f"❌ Monitor #{monitor_id} not found")
                    
            except ValueError:
                await update.message.reply_text("❌ Invalid monitor ID. Please provide a number.")
            except Exception as e:
                logger.error(f"Error in stop command: {str(e)}")
                await update.message.reply_text(f"❌ Error stopping monitor: {str(e)}")
        
        async def list_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
            """Handle /list command."""
            try:
                chat_id = str(update.effective_chat.id)
                monitors = self.monitor_service.list_monitors('telegram', chat_id)
                
                if not monitors:
                    await update.message.reply_text("No active monitors in this chat.")
                    return
                
                response = "📋 *Active Monitors:*\n\n"
                for monitor in monitors:
                    response += f"*#{monitor['id']}* - `{monitor['subject']}` | Keywords: `{monitor['keywords']}`\n"
                    response += f"  Created: {monitor['created_at'].strftime('%Y-%m-%d %H:%M')}"
                    if monitor['last_checked']:
                        response += f" | Last checked: {monitor['last_checked'].strftime('%Y-%m-%d %H:%M')}"
                    response += f" | Papers found: {monitor['paper_count']}\n\n"
                
                await update.message.reply_text(response, parse_mode=ParseMode.MARKDOWN)
                
            except Exception as e:
                logger.error(f"Error in list command: {str(e)}")
                await update.message.reply_text(f"❌ Error listing monitors: {str(e)}")
        
        async def summarize_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
            """Handle /summarize command."""
            try:
                if not context.args:
                    await update.message.reply_text(
                        "Usage: `/summarize <arxiv_url_or_id_or_doi>`\n"
                        "Example: `/summarize https://arxiv.org/abs/2301.12345`",
                        parse_mode=ParseMode.MARKDOWN
                    )
                    return
                
                paper_id = ' '.join(context.args)
                chat_id = str(update.effective_chat.id)
                
                # Send "generating" message
                processing_msg = await update.message.reply_text("🔄 Generating summary... This may take a moment.")
                
                # Get configured AI provider for this chat
                provider_name = self.monitor_service.get_bot_config(
                    'telegram', chat_id, 'ai_provider'
                )
                
                paper, summary = self.monitor_service.generate_summary(
                    paper_id, provider_name
                )
                
                # Delete processing message and send result
                await processing_msg.delete()
                response = self._format_paper_summary(paper, summary)
                await update.message.reply_text(response, parse_mode=ParseMode.MARKDOWN, disable_web_page_preview=True)
                
            except Exception as e:
                logger.error(f"Error in summarize command: {str(e)}")
                await update.message.reply_text(f"❌ Error generating summary: {str(e)}")
        
        async def config_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
            """Handle /config command."""
            try:
                if len(context.args) < 2:
                    providers = self.ai_factory.list_providers()
                    await update.message.reply_text(
                        f"Usage: `/config provider <provider_name>`\n"
                        f"Available providers: {', '.join(providers)}",
                        parse_mode=ParseMode.MARKDOWN
                    )
                    return
                
                if context.args[0] == 'provider':
                    provider_name = context.args[1].lower()
                    available_providers = self.ai_factory.list_providers()
                    
                    if provider_name not in available_providers:
                        await update.message.reply_text(
                            f"❌ Unknown provider. Available: {', '.join(available_providers)}"
                        )
                        return
                    
                    chat_id = str(update.effective_chat.id)
                    self.monitor_service.set_bot_config(
                        'telegram', chat_id, 'ai_provider', provider_name
                    )
                    
                    await update.message.reply_text(f"✅ Set AI provider to `{provider_name}` for this chat", parse_mode=ParseMode.MARKDOWN)
                else:
                    await update.message.reply_text("❌ Unknown config option. Use `provider` to set AI provider.")
                    
            except Exception as e:
                logger.error(f"Error in config command: {str(e)}")
                await update.message.reply_text(f"❌ Error updating config: {str(e)}")
        
        # Register command handlers
        self.application.add_handler(CommandHandler("start", start_command))
        self.application.add_handler(CommandHandler("help", help_command))
        self.application.add_handler(CommandHandler("monitor", monitor_command))
        self.application.add_handler(CommandHandler("stop", stop_command))
        self.application.add_handler(CommandHandler("list", list_command))
        self.application.add_handler(CommandHandler("summarize", summarize_command))
        self.application.add_handler(CommandHandler("config", config_command))
    
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
        if platform != 'telegram':
            return
        
        import asyncio
        
        async def send_papers():
            for paper in papers[:settings.max_papers_per_check]:
                try:
                    message = self._format_paper_notification(paper)
                    await self.application.bot.send_message(
                        chat_id=channel_id,
                        text=message,
                        parse_mode=ParseMode.MARKDOWN,
                        disable_web_page_preview=True
                    )
                except Exception as e:
                    logger.error(f"Error posting paper to Telegram: {str(e)}")
        
        # Run in the event loop
        try:
            loop = asyncio.get_event_loop()
            loop.create_task(send_papers())
        except RuntimeError:
            # If no event loop is running, create one
            asyncio.run(send_papers())
    
    def _format_paper_notification(self, paper: Paper) -> str:
        """Format a paper for Telegram notification."""
        authors = ", ".join(paper.authors[:3])
        if len(paper.authors) > 3:
            authors += " et al."
        
        # Escape markdown special characters
        title = paper.title.replace('*', '\\*').replace('_', '\\_').replace('`', '\\`')
        authors = authors.replace('*', '\\*').replace('_', '\\_').replace('`', '\\`')
        
        message = f"📄 *New Paper Alert*\n\n"
        message += f"*{title}*\n"
        message += f"👥 {authors}\n"
        message += f"📂 {', '.join(paper.categories)}\n"
        message += f"📅 Published: {paper.published_date.strftime('%Y-%m-%d')}\n\n"
        message += f"📖 *Abstract:*\n{paper.abstract[:300]}{'...' if len(paper.abstract) > 300 else ''}\n\n"
        message += f"🔗 [View on ArXiv]({paper.arxiv_url}) | [Download PDF]({paper.pdf_url})"
        
        return message
    
    def _format_paper_summary(self, paper: Paper, summary: str) -> str:
        """Format a paper summary for Telegram."""
        authors = ", ".join(paper.authors[:3])
        if len(paper.authors) > 3:
            authors += " et al."
        
        # Escape markdown special characters
        title = paper.title.replace('*', '\\*').replace('_', '\\_').replace('`', '\\`')
        authors = authors.replace('*', '\\*').replace('_', '\\_').replace('`', '\\`')
        
        message = f"📄 *Paper Summary*\n\n"
        message += f"*{title}*\n"
        message += f"👥 {authors}\n"
        message += f"📂 {', '.join(paper.categories)}\n"
        message += f"📅 Published: {paper.published_date.strftime('%Y-%m-%d')}\n\n"
        message += f"🤖 *AI Summary:*\n{summary}\n\n"
        message += f"🔗 [View on ArXiv]({paper.arxiv_url}) | [Download PDF]({paper.pdf_url})"
        
        return message
    
    def start(self):
        """Start the Telegram bot."""
        logger.info("Starting Telegram bot...")
        self.application.run_polling()


def main():
    """Main entry point for the Telegram bot."""
    logging.basicConfig(level=logging.INFO)
    bot = TelegramBot()
    bot.start()


if __name__ == "__main__":
    main() 