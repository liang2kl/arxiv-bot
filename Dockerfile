# Multi-purpose Dockerfile for ArXiv Bot (Slack & Telegram)
FROM python:3.11-slim as base

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY arxiv_bot/ ./arxiv_bot/

# Copy additional files
COPY test_setup.py demo.py test_docker_setup.py test_db_path.py ./
COPY scripts/ ./scripts/

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app

# Create data directory for database with proper permissions
RUN mkdir -p /app/data && chown -R app:app /app

# Switch to non-root user
USER app

# Build argument to determine which bot to run
ARG BOT_TYPE=slack

# Health check (generic for both bots)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import arxiv_bot.core.config; print('Health check OK')" || exit 1

# Use conditional CMD based on build argument
CMD if [ "$BOT_TYPE" = "telegram" ]; then \
    python -m arxiv_bot.telegram_bot; \
    else \
    python -m arxiv_bot.slack_bot; \
    fi 