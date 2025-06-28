#!/usr/bin/env python3
"""Script to help identify all MCP servers and their tool counts."""

import json
import os
from pathlib import Path

def find_claude_config():
    """Find Claude Code configuration files."""
    possible_paths = [
        Path.home() / ".config" / "claude" / "claude_desktop_config.json",
        Path.home() / ".claude" / "claude_desktop_config.json", 
        Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json",
        Path.home() / "AppData" / "Roaming" / "Claude" / "claude_desktop_config.json",
    ]
    
    config_files = []
    for path in possible_paths:
        if path.exists():
            config_files.append(path)
    
    return config_files

def analyze_mcp_config(config_path):
    """Analyze an MCP configuration file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print(f"\n📁 Config file: {config_path}")
        
        if 'mcpServers' in config:
            servers = config['mcpServers']
            print(f"Found {len(servers)} MCP servers:")
            
            total_estimated_tools = 0
            for server_name, server_config in servers.items():
                command = server_config.get('command', 'Unknown')
                args = server_config.get('args', [])
                
                print(f"\n  🖥️  Server: {server_name}")
                print(f"     Command: {command}")
                print(f"     Args: {args}")
                
                # Try to estimate tool count based on known servers
                estimated_tools = estimate_tool_count(server_name, command, args)
                print(f"     Estimated tools: {estimated_tools}")
                total_estimated_tools += estimated_tools
            
            print(f"\n📊 Total estimated tools across all servers: {total_estimated_tools}")
            
            if total_estimated_tools > 91:
                print(f"✅ Tool 91 is likely within the combined tool set")
                print(f"🔍 Need to identify which server provides tools 91+")
            else:
                print(f"❌ Estimated tool count ({total_estimated_tools}) < 91")
        else:
            print("No MCP servers found in config")
            
    except Exception as e:
        print(f"Error reading config: {e}")

def estimate_tool_count(server_name, command, args):
    """Estimate tool count based on server type."""
    server_name_lower = server_name.lower()
    command_lower = str(command).lower()
    
    # Known tool counts for common MCP servers
    if 'hades' in server_name_lower:
        return 24  # We know HADES has 24 tools
    elif 'tree-sitter' in server_name_lower or 'tree_sitter' in command_lower:
        return 30  # Tree-sitter typically has ~30 tools
    elif 'memory' in server_name_lower:
        return 10  # Memory server typically has ~10 tools
    elif 'browser' in server_name_lower:
        return 15  # Browser tools typically have ~15 tools
    elif 'github' in server_name_lower:
        return 20  # GitHub MCP typically has ~20 tools
    elif 'ollama' in server_name_lower:
        return 10  # Ollama MCP typically has ~10 tools
    elif 'filesystem' in server_name_lower or 'file' in server_name_lower:
        return 5   # File system tools typically have ~5 tools
    else:
        return 10  # Default estimate for unknown servers

def main():
    print("🔍 Identifying MCP servers and tool distribution...")
    
    config_files = find_claude_config()
    
    if not config_files:
        print("❌ No Claude Code configuration files found")
        print("Common locations:")
        print("  - ~/.config/claude/claude_desktop_config.json")
        print("  - ~/.claude/claude_desktop_config.json")
        print("  - ~/Library/Application Support/Claude/claude_desktop_config.json")
        return
    
    for config_file in config_files:
        analyze_mcp_config(config_file)
    
    print(f"\n💡 Recommendations:")
    print(f"1. Check which MCP server provides tools around index 91")
    print(f"2. Look for servers with FastAPI/OpenAPI integration")
    print(f"3. Focus on servers that auto-generate tools from APIs")
    print(f"4. Check tree-sitter, browser-tools, or custom API servers")

if __name__ == "__main__":
    main()