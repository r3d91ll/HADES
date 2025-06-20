#!/bin/bash
#
# HADES Service Management Script
#
# Unified script to manage HADES server lifecycle.
#
# Usage:
#     ./hades-service.sh --start      Start HADES server
#     ./hades-service.sh --stop       Stop HADES server
#     ./hades-service.sh --status     Show service status
#     ./hades-service.sh --restart    Restart HADES server
#

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HADES_PORT=8595
HADES_HOST="0.0.0.0" 
PIDFILE="$SCRIPT_DIR/.hades.pid"
LOGFILE="$SCRIPT_DIR/logs/hades-service.log"
POETRY_ENV="$SCRIPT_DIR/.venv/bin/activate"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}✓${NC} $1"
}

warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

error() {
    echo -e "${RED}✗${NC} $1"
}

# Ensure logs directory exists
mkdir -p "$(dirname "$LOGFILE")"

# Check if Poetry environment exists
check_poetry_env() {
    if [[ ! -f "$POETRY_ENV" ]]; then
        error "Poetry virtual environment not found at $POETRY_ENV"
        error "Please run 'poetry install' first"
        exit 1
    fi
}

# Get PID from pidfile
get_pid() {
    if [[ -f "$PIDFILE" ]]; then
        cat "$PIDFILE"
    else
        echo ""
    fi
}

# Check if process is running
is_running() {
    local pid=$(get_pid)
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        return 0
    else
        return 1
    fi
}

# Get process start time
get_start_time() {
    local pid=$(get_pid)
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        ps -o lstart= -p "$pid" 2>/dev/null | sed 's/^ *//'
    else
        echo "Not running"
    fi
}

# Get process uptime
get_uptime() {
    local pid=$(get_pid)
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        ps -o etime= -p "$pid" 2>/dev/null | sed 's/^ *//'
    else
        echo "Not running"
    fi
}
check_port() {
    if command -v netstat >/dev/null 2>&1; then
        if netstat -tlnp 2>/dev/null | grep -q ":$HADES_PORT "; then
            return 0
        fi
    elif command -v ss >/dev/null 2>&1; then
        if ss -tlnp 2>/dev/null | grep -q ":$HADES_PORT "; then
            return 0
        fi
    elif command -v lsof >/dev/null 2>&1; then
        if lsof -i ":$HADES_PORT" >/dev/null 2>&1; then
            return 0
        fi
    else
        warning "No port checking tool available (netstat/ss/lsof)"
    fi
    return 1
}

# Start HADES server
start_server() {
    log "Starting HADES server..."
    
    # Check if already running
    if is_running; then
        warning "HADES server is already running (PID: $(get_pid))"
        return 0
    fi
    
    # Check if port is in use
    if check_port; then
        error "Port $HADES_PORT is already in use"
        netstat -tlnp 2>/dev/null | grep ":$HADES_PORT " || true
        exit 1
    fi
    
    # Check Poetry environment
    check_poetry_env
    
    # Start server in background
    log "Activating Poetry environment and starting server..."
    cd "$SCRIPT_DIR"
    
    # Start server with nohup and redirect output
    nohup bash -c "
        source '$POETRY_ENV'
        export PYTHONPATH='$SCRIPT_DIR:\$PYTHONPATH'
        exec python -m src.api.server
    " > "$LOGFILE" 2>&1 &
    
    local bash_pid=$!
    
    # Wait a moment for the Python process to start
    sleep 2
    
    # Find the actual Python process (child of the bash process)
    local server_pid=$(pgrep -P "$bash_pid" -f "python.*src.api.server" | head -1)
    
    # If we couldn't find the child, use the bash PID
    if [[ -z "$server_pid" ]]; then
        server_pid=$bash_pid
    fi
    
    echo "$server_pid" > "$PIDFILE"
    
    # Wait a moment and check if it started successfully
    sleep 3
    
    if is_running; then
        success "HADES server started successfully (PID: $server_pid)"
        success "Server listening on http://$HADES_HOST:$HADES_PORT"
        success "Logs: $LOGFILE"
        
        # Test health endpoint
        log "Testing server health..."
        if curl -s "http://localhost:$HADES_PORT/health" > /dev/null; then
            success "Health check passed"
        else
            warning "Health check failed - server may still be starting"
        fi
    else
        error "Failed to start HADES server"
        if [[ -f "$LOGFILE" ]]; then
            error "Check logs at: $LOGFILE"
            tail -10 "$LOGFILE"
        fi
        rm -f "$PIDFILE"
        exit 1
    fi
}

# Stop HADES server
stop_server() {
    log "Stopping HADES server..."
    
    local pid=$(get_pid)
    local stopped=false
    
    # First try to stop the PID from pidfile
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        log "Terminating process $pid from pidfile..."
        kill "$pid"
        
        # Wait for graceful shutdown
        local count=0
        while kill -0 "$pid" 2>/dev/null && [[ $count -lt 10 ]]; do
            sleep 1
            count=$((count + 1))
        done
        
        # Force kill if still running
        if kill -0 "$pid" 2>/dev/null; then
            warning "Process did not terminate gracefully, force killing..."
            kill -9 "$pid"
            sleep 1
        fi
        
        if ! kill -0 "$pid" 2>/dev/null; then
            stopped=true
        fi
    elif [[ -n "$pid" ]]; then
        warning "Process $pid from pidfile is not running"
    else
        warning "No PID file found"
    fi
    
    # Clean up PID file
    rm -f "$PIDFILE"
    
    # Check if port is still in use and kill any process using it
    if check_port; then
        warning "Port $HADES_PORT is still in use, finding and stopping process..."
        
        # Find the actual PID using the port
        local port_pid=$(netstat -tlnp 2>/dev/null | grep ":$HADES_PORT " | awk '{print $7}' | cut -d'/' -f1)
        
        if [[ -n "$port_pid" ]]; then
            log "Found process $port_pid using port $HADES_PORT"
            
            # Check if it's a Python process running our server
            if ps -p "$port_pid" -o cmd= 2>/dev/null | grep -q "src.api.server"; then
                log "Terminating HADES process $port_pid..."
                kill "$port_pid" 2>/dev/null || true
                
                # Wait for it to stop
                local count=0
                while kill -0 "$port_pid" 2>/dev/null && [[ $count -lt 10 ]]; do
                    sleep 1
                    ((count++))
                done
                
                # Force kill if needed
                if kill -0 "$port_pid" 2>/dev/null; then
                    warning "Force killing process $port_pid..."
                    kill -9 "$port_pid" 2>/dev/null || true
                    sleep 1
                fi
                
                if ! kill -0 "$port_pid" 2>/dev/null; then
                    success "Stopped process $port_pid using port $HADES_PORT"
                    stopped=true
                else
                    error "Failed to stop process $port_pid"
                fi
            else
                error "Port $HADES_PORT is being used by a non-HADES process"
                netstat -tlnp 2>/dev/null | grep ":$HADES_PORT " || true
            fi
        fi
    fi
    
    if [[ "$stopped" == "true" ]]; then
        success "HADES server stopped successfully"
    else
        if check_port; then
            warning "Port still in use but could not stop process"
        else
            success "No HADES server was running"
        fi
    fi
}

# Show server status
show_status() {
    log "HADES Service Status"
    echo "===================="
    
    local pid=$(get_pid)
    
    if is_running; then
        success "Status: RUNNING"
        echo "PID: $pid"
        echo "Started: $(get_start_time)"
        echo "Uptime: $(get_uptime)"
        echo "Port: $HADES_PORT"
        
        # Check memory usage
        if command -v ps > /dev/null; then
            local mem_usage=$(ps -o rss= -p "$pid" 2>/dev/null | sed 's/^ *//')
            if [[ -n "$mem_usage" ]]; then
                echo "Memory: $(( mem_usage / 1024 )) MB"
            fi
        fi
        
        # Test connectivity
        echo -n "Health Check: "
        if curl -s --connect-timeout 5 "http://localhost:$HADES_PORT/health" > /dev/null; then
            success "HEALTHY"
        else
            error "UNHEALTHY"
        fi
        
    else
        error "Status: STOPPED"
        if [[ -n "$pid" ]]; then
            echo "Last PID: $pid (process not found)"
        else
            echo "No PID file found"
        fi
    fi
    
    # Show port status
    echo -n "Port $HADES_PORT: "
    if check_port; then
        local port_info=$(netstat -tlnp 2>/dev/null | grep ":$HADES_PORT " | head -1)
        echo "IN USE"
        if [[ -n "$port_info" ]]; then
            echo "  $port_info"
        fi
    else
        echo "AVAILABLE"
    fi
    
    # Show log file info
    if [[ -f "$LOGFILE" ]]; then
        echo "Log file: $LOGFILE"
        echo "Log size: $(du -h "$LOGFILE" | cut -f1)"
        echo "Recent log entries:"
        tail -5 "$LOGFILE" | while read -r line; do
            echo "  $line"
        done
    else
        echo "Log file: Not found"
    fi
}

# Restart server
restart_server() {
    log "Restarting HADES server..."
    
    # Temporarily disable exit on error for stop_server
    set +e
    stop_server
    local stop_result=$?
    set -e
    
    # Wait for clean shutdown
    log "Waiting for clean shutdown..."
    sleep 5
    
    # Make sure port is available before starting
    if check_port; then
        warning "Port $HADES_PORT still in use, waiting..."
        sleep 3
    fi
    
    log "Starting server..."
    start_server
}

# Show usage
show_usage() {
    echo "HADES Service Management Script"
    echo ""
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  --start     Start HADES server"
    echo "  --stop      Stop HADES server"  
    echo "  --restart   Restart HADES server"
    echo "  --status    Show service status"
    echo "  --help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --start          # Start the server"
    echo "  $0 --status         # Check server status"
    echo "  $0 --stop           # Stop the server"
    echo ""
    echo "Logs are written to: $LOGFILE"
    echo "PID file location: $PIDFILE"
}

# Main script logic
case "${1:-}" in
    --start)
        start_server
        ;;
    --stop)
        stop_server
        ;;
    --restart)
        restart_server
        ;;
    --status)
        show_status
        ;;
    --help|help|-h)
        show_usage
        ;;
    "")
        error "No option provided"
        echo ""
        show_usage
        exit 1
        ;;
    *)
        error "Unknown option: $1"
        echo ""
        show_usage
        exit 1
        ;;
esac