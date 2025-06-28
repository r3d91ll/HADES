#!/usr/bin/env python3
"""Dump a specific tool's schema for detailed analysis."""

import requests
import json
import sys

def dump_tool_schema(tool_name, base_url="http://localhost:8595"):
    """Dump the schema for a specific HADES tool."""
    try:
        print(f"Getting schema for tool '{tool_name}' from {base_url}...")
        
        # Get debug info about MCP tools
        debug_response = requests.get(f"{base_url}/debug/mcp-tools", timeout=10)
        if debug_response.status_code != 200:
            print(f"❌ Debug endpoint failed: {debug_response.status_code}")
            return
        
        debug_data = debug_response.json()
        
        # Find the specific tool
        tool_found = False
        for tool in debug_data.get('all_tools', []):
            if tool['name'] == tool_name:
                tool_found = True
                print(f"\n🔍 Tool: {tool['name']}")
                print(f"HADES Index: {tool['hades_index']}")
                print(f"Description: {tool['description']}")
                print(f"Has Input Schema: {tool['has_input_schema']}")
                print(f"Schema Hash: {tool['schema_hash']}")
                
                if tool['schema_issues']:
                    print(f"❌ Schema Issues:")
                    for issue in tool['schema_issues']:
                        print(f"  - {issue}")
                else:
                    print(f"✅ No schema issues detected")
                
                if tool['input_schema']:
                    print(f"\n📋 Input Schema:")
                    print(json.dumps(tool['input_schema'], indent=2))
                else:
                    print(f"\n❌ No input schema available")
                
                break
        
        if not tool_found:
            print(f"❌ Tool '{tool_name}' not found")
            print(f"Available tools:")
            for tool in debug_data.get('all_tools', []):
                print(f"  - {tool['name']}")
    
    except requests.exceptions.ConnectionError:
        print(f"❌ Could not connect to server at {base_url}")
    except Exception as e:
        print(f"❌ Error: {e}")

def list_common_tools():
    """List some common HADES tools that might be problematic."""
    common_tools = [
        'start_training',     # Complex ML parameters
        'configure_engine',   # Complex configuration
        'run_pipeline',       # Multiple optional parameters
        'start_experiment',   # Experiment configuration
        'retrieve_with_engine' # Query parameters
    ]
    
    print("🔍 Checking common tools that might have schema issues...")
    for tool in common_tools:
        print(f"\n" + "="*50)
        dump_tool_schema(tool)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        tool_name = sys.argv[1]
        dump_tool_schema(tool_name)
    else:
        print("Usage: python dump_tool_schema.py <tool_name>")
        print("Or run without arguments to check common tools:")
        print()
        list_common_tools()