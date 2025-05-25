"""Runtime package for managing LLMs via a Unix-domain-socket service.

Other modules should import `ModelClient` to interact with the running
service.  If the service socket does not yet exist, `ModelClient` will
start it automatically in a background daemon process for developer
convenience (this behaviour can be disabled in production via
`HADES_RUNTIME_AUTOSTART=0`).
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union, List, TypeVar, cast

_DEFAULT_SOCKET_PATH = os.getenv("HADES_MODEL_MGR_SOCKET", "/tmp/hades_model_mgr.sock")


def _ensure_server(socket_path: Optional[str] = _DEFAULT_SOCKET_PATH) -> None:
    """Spawn the model-manager server process if the socket is missing.

    This helper is intentionally very small to avoid import-heavy
    dependencies at interpreter start-up.  It uses ``subprocess`` to
    launch ``python -m src.runtime.server`` as a daemon.
    """
    import sys
    # Handle None case with default value
    actual_socket_path = socket_path if socket_path is not None else _DEFAULT_SOCKET_PATH
    
    if os.path.exists(actual_socket_path):
        return

    # Check at runtime to respect env var changes after module load
    if os.getenv("HADES_RUNTIME_AUTOSTART", "1") == "0":
        raise RuntimeError(
            f"Model-manager socket not found and autostart disabled for {socket_path}. "
            "Start the server via `python -m src.runtime.server`.")

    print(f"[ModelClient] Attempting to autostart model manager server at {actual_socket_path}", file=sys.stderr)
    python_exe = sys.executable
    # Ensure all elements in the command list are strings
    server_module = "src.model_engine.engines.haystack.runtime.server"
    # Cast to str to ensure mypy knows it's a string
    socket_arg = str(actual_socket_path)
    cmd = [python_exe, "-m", server_module, socket_arg]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       start_new_session=True)
        # Check if process exited immediately (likely an error)
        if proc.poll() is not None:
            print(f"[ModelClient] Server process exited immediately with code {proc.returncode}", file=sys.stderr)
    except Exception as e:
        print(f"[ModelClient] Failed to launch server: {e}", file=sys.stderr)
        raise

    for _ in range(30):
        if os.path.exists(actual_socket_path):
            return
        time.sleep(0.1)

    raise RuntimeError(f"Failed to start model-manager server; socket never appeared at {actual_socket_path}.")


class ModelClient:
    """Lightweight JSON-RPC client that talks to the UDS model manager."""

    def __init__(self, socket_path: Optional[str] = None) -> None:
        # Allow override via constructor, but default to env var
        self.socket_path = socket_path or _DEFAULT_SOCKET_PATH
        # Ensure server is running (noop if already running)
        _ensure_server(self.socket_path)

    # ---------------------------------------------------------------------
    # Public API mirrors the actions supported by the server.
    # ---------------------------------------------------------------------
    def ping(self) -> str:
        result = self._request({"action": "ping"})["result"]
        assert isinstance(result, str)
        return result

    def load(self, model_id: str, device: Optional[str] = None) -> str:
        result = self._request({"action": "load", "model_id": model_id, "device": device})["result"]
        assert isinstance(result, str)
        return result

    def unload(self, model_id: str) -> str:
        result = self._request({"action": "unload", "model_id": model_id})["result"]
        assert isinstance(result, str)
        return result
        
    def info(self) -> Dict[str, Any]:
        """Get information about loaded models from the server.
        
        Returns:
            Dictionary of model information
            
        Raises:
            RuntimeError: If the server returns an error
        """
        response = self._request({"action": "info"})
        print(f"[ModelClient] Info response: {response}")
        
        if "error" in response:
            raise RuntimeError(f"Server error: {response['error']}")
        
        if "result" not in response:
            raise RuntimeError(f"Invalid response from server: {response}")
            
        result = response["result"]
        assert isinstance(result, dict)
        return result
        
    def debug(self) -> Dict[str, Any]:
        """Get detailed debug information about the model server.
        
        Returns:
            Dictionary with detailed debug information
            
        Raises:
            RuntimeError: If the server returns an error
        """
        response = self._request({"action": "debug"})
        print(f"[ModelClient] Debug response: {response}")
        
        if "error" in response:
            raise RuntimeError(f"Server error: {response['error']}")
        
        if "result" not in response:
            raise RuntimeError(f"Invalid response from server: {response}")
            
        result = response["result"]
        assert isinstance(result, dict)
        return result
        
    def shutdown(self) -> str:
        """Request the server to shut down gracefully.
        
        This will cause the server process to exit, cleaning up all resources including
        loaded models. After calling this, the client will no longer be able to communicate
        with the server until it's started again.
        
        Returns:
            Status message confirming the shutdown was initiated
            
        Raises:
            RuntimeError: If the server returns an error
        """
        try:
            response = self._request({"action": "shutdown"})
            print(f"[ModelClient] Shutdown response: {response}")
            
            if "error" in response:
                raise RuntimeError(f"Server error: {response['error']}")
            
            if "result" not in response:
                raise RuntimeError(f"Invalid response from server: {response}")
                
            result = response["result"]
            assert isinstance(result, str)
            return result
        except (ConnectionRefusedError, ConnectionResetError):
            # The server might shut down quickly, so these errors are expected
            print("[ModelClient] Connection closed during shutdown - this is expected")
            return "shutdown_in_progress"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        import socket, json
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.connect(self.socket_path)
                sock.sendall(json.dumps(payload).encode())
                data = sock.recv(65536)
                if not data:
                    raise RuntimeError("No response from model manager server (connection closed)")
        except ConnectionResetError as e:
            raise RuntimeError(f"Connection to model manager server reset: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to communicate with model manager server: {e}") from e
        obj = json.loads(data.decode())
        assert isinstance(obj, dict)
        return obj


from .server import (
    run_server,
    _load_model,
    _unload_model,
    _get_model_info,
    _debug_cache,
    _shutdown_server,
    _handle_request,
    _handle_conn,
    _is_server_running,
    socket,
)

__all__ = [
    "ModelClient",
    "run_server",
    "_load_model",
    "_unload_model",
    "_get_model_info",
    "_debug_cache",
    "_shutdown_server",
    "_handle_request",
    "_handle_conn",
    "_is_server_running",
    "socket",
]

