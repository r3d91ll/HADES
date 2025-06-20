#!/bin/bash

# HADES Agent Management Script
# 
# This script provides an easy CLI interface for managing OAuth agents
# in the HADES minimal OAuth 2.0 system.
#
# ⚠️  WARNING: LAB ENVIRONMENT ONLY ⚠️

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON_CMD="poetry run python"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

show_usage() {
    cat << EOF
HADES Agent Management Tool

Usage: $0 <command> [arguments]

Commands:
    create <agent_id> <name> [description]   Create a new agent and token
    list                                     List all agents
    info <agent_id>                         Show detailed agent information
    token <agent_id>                        Generate new token for agent
    revoke <agent_id>                       Revoke all tokens for agent
    deactivate <agent_id>                   Deactivate agent
    cleanup                                 Remove expired tokens
    help                                    Show this help message

Examples:
    $0 create claude-agent-1 "Claude Agent" "Agent for testing"
    $0 list
    $0 token claude-agent-1
    $0 revoke claude-agent-1

⚠️  WARNING: This is for LAB ENVIRONMENTS ONLY!
For production use, implement proper OAuth 2.0 with enterprise providers.
EOF
}

run_python_script() {
    cd "$PROJECT_ROOT" || { log_error "Failed to change to project root: $PROJECT_ROOT"; exit 1; }
    $PYTHON_CMD -c "$1"
}

create_agent() {
    local agent_id="$1"
    local name="$2"
    local description="${3:-}"
    
    if [[ -z "$agent_id" || -z "$name" ]]; then
        log_error "Usage: $0 create <agent_id> <name> [description]"
        exit 1
    fi
    
    log_info "Creating agent: $agent_id ($name)"
    
    # Python script to create agent and token
    local python_script="
import sys
import json
sys.path.insert(0, '.')
from src.auth.token_manager import TokenManager

# Parse arguments safely
agent_id = sys.argv[1]
name = sys.argv[2]
description = sys.argv[3] if len(sys.argv) > 3 else ''

try:
    tm = TokenManager()
    
    # Create agent
    agent = tm.create_agent(agent_id, name, description)
    print(f'✓ Created agent: {agent.agent_id}')
"
    
    cd "$PROJECT_ROOT"
    $PYTHON_CMD -c "$python_script" "$agent_id" "$name" "$description"
    
    # Create initial token
    token = tm.create_token('$agent_id', expires_in_days=30)
    print(f'✓ Generated token (expires in 30 days)')
    print()
    print('=== AGENT CONFIGURATION ===')
    print(f'Agent ID: {agent.agent_id}')
    print(f'Name: {agent.name}')
    print(f'Description: {agent.description}')
    print(f'Token: {token}')
    print()
    print('=== MCP CLIENT CONFIGURATION ===')
    print('Add this to your MCP client configuration:')
    print('{')
    print(f'  \"mcpServers\": {{')
    print(f'    \"hades-{agent.agent_id}\": {{')
    print(f'      \"command\": \"npx\",')
    print(f'      \"args\": [\"@modelcontextprotocol/server-fetch\"],')
    print(f'      \"env\": {{')
    print(f'        \"FETCH_BASE_URL\": \"http://localhost:8595\",')
    print(f'        \"FETCH_API_KEY\": \"{token}\"')
    print(f'      }}')
    print(f'    }}')
    print(f'  }}')
    print('}')
    print()
    print('⚠️  SECURITY WARNING: Store this token securely!')
    print('⚠️  This setup is for LAB ENVIRONMENTS ONLY!')
    
except ValueError as e:
    print(f'ERROR: {e}')
    sys.exit(1)
except Exception as e:
    print(f'UNEXPECTED ERROR: {e}')
    sys.exit(1)
"
    
    run_python_script "$python_script"
    log_success "Agent created successfully!"
}

list_agents() {
    log_info "Listing all agents..."
    
    local python_script="
import sys
sys.path.insert(0, '.')
from src.auth.token_manager import TokenManager
from datetime import datetime

try:
    tm = TokenManager()
    agents = tm.list_agents()
    
    if not agents:
        print('No agents found.')
        sys.exit(0)
    
    print(f'Found {len(agents)} agent(s):')
    print()
    
    for agent in agents:
        status = '🟢 ACTIVE' if agent.is_active else '🔴 INACTIVE'
        last_used = agent.last_used.strftime('%Y-%m-%d %H:%M:%S') if agent.last_used else 'Never'
        
        print(f'Agent ID: {agent.agent_id}')
        print(f'  Name: {agent.name}')
        print(f'  Status: {status}')
        print(f'  Created: {agent.created_at.strftime(\"%Y-%m-%d %H:%M:%S\")}')
        print(f'  Last Used: {last_used}')
        if agent.description:
            print(f'  Description: {agent.description}')
        print()
        
except Exception as e:
    print(f'ERROR: {e}')
    sys.exit(1)
"
    
    run_python_script "$python_script"
}

show_agent_info() {
    local agent_id="$1"
    
    if [[ -z "$agent_id" ]]; then
        log_error "Usage: $0 info <agent_id>"
        exit 1
    fi
    
    log_info "Getting information for agent: $agent_id"
    
    local python_script="
import sys
sys.path.insert(0, '.')
from src.auth.token_manager import TokenManager

try:
    tm = TokenManager()
    info = tm.get_agent_info('$agent_id')
    
    if not info:
        print('Agent not found.')
        sys.exit(1)
    
    status = '🟢 ACTIVE' if info['is_active'] else '🔴 INACTIVE'
    last_used = info['last_used'] if info['last_used'] else 'Never'
    
    print(f'=== AGENT INFORMATION ===')
    print(f'Agent ID: {info[\"agent_id\"]}')
    print(f'Name: {info[\"name\"]}')
    print(f'Status: {status}')
    print(f'Created: {info[\"created_at\"]}')
    print(f'Last Used: {last_used}')
    print(f'Active Tokens: {info[\"active_tokens\"]}')
    if info['description']:
        print(f'Description: {info[\"description\"]}')
        
except Exception as e:
    print(f'ERROR: {e}')
    sys.exit(1)
"
    
    run_python_script "$python_script"
}

generate_token() {
    local agent_id="$1"
    
    if [[ -z "$agent_id" ]]; then
        log_error "Usage: $0 token <agent_id>"
        exit 1
    fi
    
    log_info "Generating new token for agent: $agent_id"
    
    local python_script="
import sys
sys.path.insert(0, '.')
from src.auth.token_manager import TokenManager

try:
    tm = TokenManager()
    
    # Verify agent exists
    info = tm.get_agent_info('$agent_id')
    if not info:
        print('Agent not found.')
        sys.exit(1)
    
    if not info['is_active']:
        print('Agent is inactive. Cannot generate token.')
        sys.exit(1)
    
    # Generate token
    token = tm.create_token('$agent_id', expires_in_days=30)
    
    print(f'✓ Generated new token for {info[\"name\"]} ({info[\"agent_id\"]})')
    print(f'Token: {token}')
    print(f'Expires: 30 days from now')
    print()
    print('⚠️  SECURITY WARNING: Store this token securely!')
    print('⚠️  This token provides API access to HADES!')
    
except ValueError as e:
    print(f'ERROR: {e}')
    sys.exit(1)
except Exception as e:
    print(f'UNEXPECTED ERROR: {e}')
    sys.exit(1)
"
    
    run_python_script "$python_script"
    log_success "Token generated successfully!"
}

revoke_agent() {
    local agent_id="$1"
    
    if [[ -z "$agent_id" ]]; then
        log_error "Usage: $0 revoke <agent_id>"
        exit 1
    fi
    
    log_warning "This will revoke ALL tokens for agent: $agent_id"
    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Operation cancelled."
        exit 0
    fi
    
    local python_script="
import sys
sys.path.insert(0, '.')
from src.auth.token_manager import TokenManager

try:
    tm = TokenManager()
    count = tm.revoke_agent_tokens('$agent_id')
    print(f'✓ Revoked {count} token(s) for agent $agent_id')
    
except Exception as e:
    print(f'ERROR: {e}')
    sys.exit(1)
"
    
    run_python_script "$python_script"
    log_success "Tokens revoked successfully!"
}

deactivate_agent() {
    local agent_id="$1"
    
    if [[ -z "$agent_id" ]]; then
        log_error "Usage: $0 deactivate <agent_id>"
        exit 1
    fi
    
    log_warning "This will DEACTIVATE agent and revoke ALL tokens: $agent_id"
    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Operation cancelled."
        exit 0
    fi
    
    local python_script="
import sys
sys.path.insert(0, '.')
from src.auth.token_manager import TokenManager

try:
    tm = TokenManager()
    success = tm.deactivate_agent('$agent_id')
    
    if success:
        print(f'✓ Deactivated agent $agent_id and revoked all tokens')
    else:
        print(f'Agent $agent_id not found')
        sys.exit(1)
    
except Exception as e:
    print(f'ERROR: {e}')
    sys.exit(1)
"
    
    run_python_script "$python_script"
    log_success "Agent deactivated successfully!"
}

cleanup_tokens() {
    log_info "Cleaning up expired tokens..."
    log_warning "This feature is not yet implemented."
    # TODO: Implement expired token cleanup
}

# Main script logic
main() {
    if [[ $# -eq 0 ]]; then
        show_usage
        exit 1
    fi
    
    local command="$1"
    shift
    
    case "$command" in
        "create")
            create_agent "$@"
            ;;
        "list")
            list_agents
            ;;
        "info")
            show_agent_info "$@"
            ;;
        "token")
            generate_token "$@"
            ;;
        "revoke")
            revoke_agent "$@"
            ;;
        "deactivate")
            deactivate_agent "$@"
            ;;
        "cleanup")
            cleanup_tokens
            ;;
        "help"|"-h"|"--help")
            show_usage
            ;;
        *)
            log_error "Unknown command: $command"
            echo
            show_usage
            exit 1
            ;;
    esac
}

# Run main function
main "$@"