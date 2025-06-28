# Fix for Claude TUI MCP System Message Error

## Problem
The Claude TUI is showing formatting errors when handling MCP (Model Context Protocol) system messages, displaying raw message objects instead of properly formatted text.

## Potential Solutions

### 1. Update claude-code-sdk
The issue might be a version mismatch. Try updating:

```bash
# In the claude-tui-editor directory
cd /home/todd/ML-Lab/claude_code_idea/claude-tui-editor
pip install --upgrade claude-code-sdk
```

### 2. Disable Problematic MCP Servers
Temporarily disable MCP servers to isolate the issue. Edit `~/.config/claude-code/config.json` and set `"disabled": true` for each server:

```json
{
  "mcpServers": {
    "tree_sitter": {
      "disabled": true,
      // ... rest of config
    },
    // ... disable other servers similarly
  }
}
```

### 3. Check MCP Server Compatibility
Some MCP servers might be sending messages in formats the TUI doesn't expect. Try running with only one MCP server enabled at a time to identify which one is causing issues.

### 4. Modify Message Handling in claude_integration.py
The TUI's message handling might need to be updated to handle SystemMessage objects. In `claude_integration.py`, around line 48-65, the message handling logic might need to be enhanced to handle more message types:

```python
# Add handling for SystemMessage type
if hasattr(message, '__class__') and message.__class__.__name__ == 'SystemMessage':
    # Extract content from SystemMessage
    if hasattr(message, 'content'):
        content = str(message.content)
        response_text += content
        self._update_response(response_text, start_pos)
    continue
```

### 5. Use Official Claude CLI Instead
If the custom TUI continues to have issues, consider using the official Claude CLI:

```bash
claude chat
```

### 6. Check for Updates to Ollama MCP
The error mentions "Ollama-mcp" which isn't in your current config. If you recently added an Ollama MCP server, ensure it's properly configured and compatible with the claude-code-sdk version.

## Debugging Steps

1. **Check claude-code-sdk version**:
   ```bash
   pip show claude-code-sdk
   ```

2. **Test with minimal MCP config**:
   - Backup current config: `cp ~/.config/claude-code/config.json ~/.config/claude-code/config.json.bak`
   - Create minimal config with no MCP servers
   - Test if TUI works without MCP servers

3. **Check for error logs**:
   ```bash
   # Check for any error logs
   ls -la ~/.cache/claude-cli-nodejs/
   ```

4. **Run in debug mode** (if available):
   ```bash
   # Check if there's a debug flag
   python /home/todd/ML-Lab/claude_code_idea/claude-tui-editor/main.py --debug
   ```

## Alternative Solution: Simple Claude Interface
If the TUI continues to have issues, you can create a simple Python script to interact with Claude without the TUI complexity:

```python
#!/usr/bin/env python3
import subprocess
import sys

def claude_chat(message):
    try:
        result = subprocess.run(
            ['claude', 'chat', message],
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        claude_chat(" ".join(sys.argv[1:]))
    else:
        print("Usage: python simple_claude.py 'your message here'")
```

Save this as `simple_claude.py` and use it as a workaround while debugging the TUI issues.