# ArXiv Bot

> [!IMPORTANT]
> The entire codebase, including documentation and scripts, is created by AI (`claude-4-sonnet`). The logs of the development process are in [logs.md](logs.md). Use it at your own risk.

A flexible bot system for Slack and Telegram that monitors ArXiv papers and provides AI-powered summaries.

## Features

1. **Paper Monitoring**: Periodically search ArXiv by subject and keywords, automatically posting new papers to channels
2. **AI Summaries**: Generate descriptive summaries of papers using OpenAI, Google Gemini, or Anthropic Claude
3. **Multi-platform**: Support for both Slack and Telegram
4. **Configurable**: Easy setup with environment variables and channel commands

## Setup

### Option 1: Local Development
1. Install dependencies:
```bash
python3 -m venv venv
source venv/bin/activate  # or venv/bin/activate.fish for fish shell
pip install -r requirements.txt
```

2. Copy `.env.example` to `.env` and configure your API keys:
```bash
cp .env.example .env
# Edit .env with your actual API keys
```

3. Initialize the database:
```bash
python -m arxiv_bot.core.database init
```

### Option 2: Docker Deployment (Recommended)
1. Install Docker and Docker Compose
2. Copy `.env.example` to `.env` and configure your API keys
3. Deploy with Docker Compose:
```bash
mkdir -p data logs
docker-compose up -d
```

See [DOCKER.md](DOCKER.md) for detailed Docker deployment instructions.

## Configuration

### Environment Variables

- `OPENAI_API_KEY`: OpenAI API key (optional)
- `GOOGLE_API_KEY`: Google Gemini API key (optional)
- `ANTHROPIC_API_KEY`: Anthropic Claude API key (optional)
- `DEFAULT_AI_PROVIDER`: Default AI provider (openai, google, anthropic)
- `SLACK_BOT_TOKEN`: Slack bot token (for Slack bot)
- `SLACK_APP_TOKEN`: Slack app token (for Slack bot)
- `TELEGRAM_BOT_TOKEN`: Telegram bot token (for Telegram bot)

### Running the Bots

#### Local Development
```bash
# Slack Bot
python -m arxiv_bot.slack_bot

# Telegram Bot
python -m arxiv_bot.telegram_bot
```

#### Docker
```bash
# Both bots with Docker Compose
docker-compose up -d

# Individual bots
docker run -d --name arxiv-bot-slack --env-file .env -v $(pwd)/data:/app/data arxiv-bot-slack
docker run -d --name arxiv-bot-telegram --env-file .env -v $(pwd)/data:/app/data arxiv-bot-telegram
```

## Usage

### Commands

- `/arxiv_monitor <subject> <keywords>`: Start monitoring papers for given subject and keywords
- `/arxiv_stop <monitor_id>`: Stop a specific monitor
- `/arxiv_list`: List all active monitors
- `/arxiv_summarize <arxiv_url_or_doi>`: Get AI summary of a paper
- `/arxiv_config provider <provider_name>`: Set default AI provider

### Subjects

Common ArXiv subjects include:
- `cs.AI` - Artificial Intelligence
- `cs.LG` - Machine Learning
- `cs.CV` - Computer Vision
- `physics.data-an` - Data Analysis
- `stat.ML` - Statistics Machine Learning

## Architecture

- `arxiv_bot.core`: Shared functionality (ArXiv API, AI providers, database)
- `arxiv_bot.slack_bot`: Slack-specific implementation
- `arxiv_bot.telegram_bot`: Telegram-specific implementation 