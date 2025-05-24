#!/bin/bash
# Docker Helper Script for ArXiv Bot

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if .env exists
check_env() {
    if [ ! -f .env ]; then
        print_error ".env file not found!"
        print_info "Please copy .env.example to .env and configure your API keys"
        print_info "cp .env.example .env"
        exit 1
    fi
}

# Function to create required directories
setup_dirs() {
    print_info "Creating required directories..."
    mkdir -p data logs
    print_success "Directories created"
}

# Function to build images
build() {
    print_info "Building Docker images..."
    
    if [ "$1" = "slack" ]; then
        docker build -f Dockerfile.slack -t arxiv-bot-slack .
        print_success "Slack bot image built"
    elif [ "$1" = "telegram" ]; then
        docker build -f Dockerfile.telegram -t arxiv-bot-telegram .
        print_success "Telegram bot image built"
    else
        docker-compose build
        print_success "All images built"
    fi
}

# Function to start services
start() {
    check_env
    setup_dirs
    
    print_info "Starting ArXiv Bot services..."
    
    # Ensure database is initialized first
    if [ ! -f data/arxiv_bot.db ]; then
        print_info "Database not found, initializing..."
        init_db
    fi
    
    if [ "$1" = "slack" ]; then
        docker-compose up -d arxiv-bot-slack
        print_success "Slack bot started"
    elif [ "$1" = "telegram" ]; then
        docker-compose up -d arxiv-bot-telegram
        print_success "Telegram bot started"
    else
        docker-compose up -d arxiv-bot-slack arxiv-bot-telegram
        print_success "All services started"
    fi
}

# Function to stop services
stop() {
    print_info "Stopping ArXiv Bot services..."
    docker-compose down
    print_success "Services stopped"
}

# Function to restart services
restart() {
    print_info "Restarting ArXiv Bot services..."
    stop
    start "$1"
}

# Function to show logs
logs() {
    if [ "$1" = "slack" ]; then
        docker-compose logs -f arxiv-bot-slack
    elif [ "$1" = "telegram" ]; then
        docker-compose logs -f arxiv-bot-telegram
    else
        docker-compose logs -f
    fi
}

# Function to show status
status() {
    print_info "Service status:"
    docker-compose ps
}

# Function to initialize database
init_db() {
    print_info "Initializing database..."
    setup_dirs
    
    # Remove existing init container if it exists
    docker-compose rm -f arxiv-bot-init 2>/dev/null || true
    
    # Run database initialization
    if docker-compose run --rm arxiv-bot-init; then
        print_success "Database initialized successfully"
        
        # Verify database was created
        if [ -f data/arxiv_bot.db ]; then
            print_success "Database file created at data/arxiv_bot.db"
        else
            print_error "Database file was not created!"
            return 1
        fi
    else
        print_error "Database initialization failed!"
        return 1
    fi
}

# Function to backup database
backup_db() {
    if [ ! -f data/arxiv_bot.db ]; then
        print_error "Database file not found!"
        exit 1
    fi
    
    backup_file="data/arxiv_bot_backup_$(date +%Y%m%d_%H%M%S).db"
    cp data/arxiv_bot.db "$backup_file"
    print_success "Database backed up to $backup_file"
}

# Function to clean up
cleanup() {
    print_warning "This will remove all containers, images, and volumes!"
    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Cleaning up Docker resources..."
        docker-compose down -v
        docker rmi arxiv-bot-slack arxiv-bot-telegram 2>/dev/null || true
        docker system prune -f
        print_success "Cleanup completed"
    else
        print_info "Cleanup cancelled"
    fi
}

# Function to run tests in container
test() {
    print_info "Running tests in container..."
    docker run --rm -v $(pwd)/data:/app/data arxiv-bot-slack python test_setup.py
}

# Function to test Docker setup
test_docker() {
    print_info "Running Docker-specific tests..."
    
    # Ensure database is initialized
    if [ ! -f data/arxiv_bot.db ]; then
        print_info "Initializing database for testing..."
        init_db || return 1
    fi
    
    docker run --rm \
        -e DATABASE_URL=sqlite:////app/data/arxiv_bot.db \
        -v $(pwd)/data:/app/data \
        arxiv-bot-slack \
        python test_docker_setup.py
}

# Function to run demo in container
demo() {
    print_info "Running demo in container..."
    docker run --rm -v $(pwd)/data:/app/data arxiv-bot-slack python demo.py
}

# Help function
help() {
    echo "ArXiv Bot Docker Helper Script"
    echo ""
    echo "Usage: $0 <command> [service]"
    echo ""
    echo "Commands:"
    echo "  build [slack|telegram]  - Build Docker images"
    echo "  start [slack|telegram]  - Start services"
    echo "  stop                    - Stop all services"
    echo "  restart [slack|telegram]- Restart services"
    echo "  logs [slack|telegram]   - Show logs"
    echo "  status                  - Show service status"
    echo "  init-db                 - Initialize database"
    echo "  backup-db               - Backup database"
    echo "  test                    - Run tests in container"
    echo "  test-docker             - Run Docker-specific tests"
    echo "  demo                    - Run demo in container"
    echo "  cleanup                 - Remove all Docker resources"
    echo "  help                    - Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 start               # Start both bots"
    echo "  $0 start slack         # Start only Slack bot"
    echo "  $0 logs telegram       # Show Telegram bot logs"
    echo "  $0 backup-db           # Backup database"
}

# Main script logic
case "$1" in
    build)
        build "$2"
        ;;
    start)
        start "$2"
        ;;
    stop)
        stop
        ;;
    restart)
        restart "$2"
        ;;
    logs)
        logs "$2"
        ;;
    status)
        status
        ;;
    init-db)
        init_db
        ;;
    backup-db)
        backup_db
        ;;
    test)
        test
        ;;
    test-docker)
        test_docker
        ;;
    demo)
        demo
        ;;
    cleanup)
        cleanup
        ;;
    help|--help|-h)
        help
        ;;
    *)
        print_error "Unknown command: $1"
        echo ""
        help
        exit 1
        ;;
esac 