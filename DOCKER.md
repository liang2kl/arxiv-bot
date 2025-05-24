# ArXiv Bot Docker Deployment Guide

This guide covers deploying the ArXiv Bot using Docker containers for both Slack and Telegram platforms.

## Quick Start

1. **Clone and setup**:
   ```bash
   git clone <repository>
   cd arxiv-bot
   cp .env.example .env
   # Edit .env with your API keys
   ```

2. **Initialize database**:
   ```bash
   ./docker-helper.sh init-db
   ```

3. **Start the bots**:
   ```bash
   ./docker-helper.sh start
   ```

## Docker Helper Script

The `docker-helper.sh` script provides convenient commands for managing the Docker deployment:

```bash
# Database operations
./docker-helper.sh init-db          # Initialize database
./docker-helper.sh backup-db        # Backup database

# Container management
./docker-helper.sh build [slack|telegram]  # Build images
./docker-helper.sh start [slack|telegram]  # Start services
./docker-helper.sh stop             # Stop all services
./docker-helper.sh restart [slack|telegram] # Restart services

# Monitoring and testing
./docker-helper.sh status           # Show service status
./docker-helper.sh logs [slack|telegram]    # View logs
./docker-helper.sh test             # Run tests
./docker-helper.sh test-docker      # Run Docker-specific tests

# Maintenance
./docker-helper.sh cleanup          # Remove all Docker resources
```

## Individual Deployment

### Slack Bot Only
```bash
./docker-helper.sh build slack
./docker-helper.sh start slack
```

### Telegram Bot Only
```bash
./docker-helper.sh build telegram
./docker-helper.sh start telegram
```

## Configuration

### Environment Variables

Create a `.env` file with the following variables:

```bash
# AI Provider Configuration
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
DEFAULT_AI_PROVIDER=openai

# Slack Configuration (for Slack bot)
SLACK_BOT_TOKEN=xoxb-your-bot-token
SLACK_APP_TOKEN=xapp-your-app-token

# Telegram Configuration (for Telegram bot)
TELEGRAM_BOT_TOKEN=your_telegram_bot_token

# Database Configuration
DATABASE_URL=sqlite:///app/data/arxiv_bot.db

# Monitoring Configuration
MONITOR_INTERVAL_HOURS=6
MAX_PAPERS_PER_CHECK=5
```

### Database Setup

The database is automatically initialized when you run:
```bash
./docker-helper.sh init-db
```

This creates:
- SQLite database at `data/arxiv_bot.db`
- Required tables (monitors, tracked_papers, bot_configs)
- Proper permissions for container access

**Note**: The database initialization has been fixed to properly handle volume mounting and path resolution in Docker containers.

## Docker Compose Services

### Services Overview

- **arxiv-bot-slack**: Slack bot container
- **arxiv-bot-telegram**: Telegram bot container  
- **arxiv-bot-init**: Database initialization service (runs once)

### Shared Resources

- **Database**: Shared SQLite database volume at `./data:/app/data`
- **Logs**: Shared logs directory at `./logs:/app/logs`
- **Network**: Dedicated bridge network for service communication

### Health Checks

Both bot services include health checks that verify:
- Python environment is working
- Core modules can be imported
- Configuration is accessible

## Monitoring and Logs

### View Logs
```bash
# All services
./docker-helper.sh logs

# Specific service
./docker-helper.sh logs slack
./docker-helper.sh logs telegram
```

### Check Status
```bash
./docker-helper.sh status
```

### Test Setup
```bash
# Test basic functionality
./docker-helper.sh test

# Test Docker-specific setup
./docker-helper.sh test-docker
```

## Maintenance

### Database Backup
```bash
./docker-helper.sh backup-db
```

### Update Images
```bash
./docker-helper.sh build
./docker-helper.sh restart
```

### Clean Restart
```bash
./docker-helper.sh stop
./docker-helper.sh cleanup
./docker-helper.sh init-db
./docker-helper.sh start
```

## Troubleshooting

### Database Issues

If you encounter database-related errors:

1. **Check database file exists**:
   ```bash
   ls -la data/
   ```

2. **Reinitialize database**:
   ```bash
   rm -f data/arxiv_bot.db
   ./docker-helper.sh init-db
   ```

3. **Check permissions**:
   ```bash
   chmod 777 data/
   ```

### Container Issues

1. **Check container status**:
   ```bash
   ./docker-helper.sh status
   ```

2. **View container logs**:
   ```bash
   ./docker-helper.sh logs
   ```

3. **Rebuild containers**:
   ```bash
   ./docker-helper.sh build
   ```

### Common Problems

- **Database not found**: Run `./docker-helper.sh init-db`
- **Permission denied**: Ensure `data/` directory has write permissions
- **Import errors**: Rebuild images with `./docker-helper.sh build`
- **API key errors**: Check `.env` file configuration

## Production Deployment

### Using PostgreSQL

For production, consider using PostgreSQL instead of SQLite:

```bash
# docker-compose.prod.yml
services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: arxiv_bot
      POSTGRES_USER: arxiv_bot
      POSTGRES_PASSWORD: your_password
    volumes:
      - postgres_data:/var/lib/postgresql/data

  arxiv-bot-slack:
    environment:
      DATABASE_URL: postgresql://arxiv_bot:your_password@postgres:5432/arxiv_bot
    depends_on:
      - postgres
```

### Security Considerations

1. **Use secrets management** for API keys
2. **Enable SSL/TLS** for database connections
3. **Run containers as non-root** (already implemented)
4. **Use specific image tags** instead of `latest`
5. **Implement log rotation** for production logs

### Scaling

For high-volume deployments:

1. **Use external database** (PostgreSQL/MySQL)
2. **Implement Redis** for caching
3. **Use container orchestration** (Kubernetes)
4. **Monitor resource usage** and scale accordingly

## Files Structure

```
arxiv-bot/
├── docker-compose.yml          # Main orchestration file
├── Dockerfile                  # Multi-purpose Dockerfile
├── Dockerfile.slack           # Slack-specific Dockerfile
├── Dockerfile.telegram        # Telegram-specific Dockerfile
├── docker-helper.sh           # Management script
├── .dockerignore              # Docker build context exclusions
├── data/                      # Database storage (created automatically)
├── logs/                      # Application logs (created automatically)
└── scripts/
    └── init-db.py            # Database initialization script
```

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review container logs with `./docker-helper.sh logs`
3. Run tests with `./docker-helper.sh test-docker`
4. Check the main README.md for general setup instructions 