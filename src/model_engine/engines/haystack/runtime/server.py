"""Model Manager server – owns GPU-resident models and exposes a minimal
JSON-RPC interface over a Unix-domain socket.
"""
from __future__ import annotations

import atexit
import json
import os
import signal
import socket
import threading
import time
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Tuple, Optional, Union, List, cast


import torch
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------------
# Configuration (could later read from a yaml or env-vars)
# ---------------------------------------------------------------------------
MAX_MODELS: int = int(os.getenv("HADES_MAX_MODELS", "3"))
DEFAULT_DEVICE: str = os.getenv("HADES_DEFAULT_DEVICE", "cuda:0")
SOCKET_PATH: str = os.getenv("HADES_MODEL_MGR_SOCKET", "/tmp/hades_model_mgr.sock")


# ---------------------------------------------------------------------------
# Server status check
# ---------------------------------------------------------------------------
def _is_server_running(socket_path: Optional[str] = None) -> bool:
    """Check if the model manager server is running.
    
    This checks if the socket exists and can be connected to.
    
    Args:
        socket_path: Path to the socket file (defaults to SOCKET_PATH)
        
    Returns:
        True if the server is running and the socket is accessible, False otherwise
    """
    socket_path = socket_path or SOCKET_PATH
    if not os.path.exists(socket_path):
        return False
    
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.connect(socket_path)
            # If we can connect, the server is running
            return True
    except Exception:
        # Any error means the server is not running properly
        return False


# ---------------------------------------------------------------------------
# In-memory LRU cache
# ---------------------------------------------------------------------------
class _LRUCache:
    def __init__(self, max_size: int) -> None:
        self._max: int = max_size
        self._lock: threading.Lock = threading.Lock()
        self._data: Dict[str, Tuple[Any, Any, float]] = {}

    def get(self, key: str) -> tuple[Any, Any] | None:
        with self._lock:
            if key not in self._data:
                return None
            model, tok, _ = self._data[key]
            self._data[key] = (model, tok, time.time())
            return model, tok

    def put(self, key: str, model: Any, tok: Any) -> None:
        with self._lock:
            if key in self._data:
                self._data[key] = (model, tok, time.time())
                return
            if len(self._data) >= self._max:
                # evict oldest
                oldest = min(self._data, key=lambda k: self._data[k][2])
                self.evict(oldest)
            self._data[key] = (model, tok, time.time())

    def evict(self, key: str) -> None:
        model, _, _ = self._data.pop(key, (None, None, None))
        if model is not None:
            model.cpu()
            del model
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def info(self) -> MappingProxyType[str, float]:
        with self._lock:
            return MappingProxyType({k: v[2] for k, v in self._data.items()})


_CACHE = _LRUCache(MAX_MODELS)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _load_model(model_id: str, device: Optional[str] = None) -> str:
    device = device or DEFAULT_DEVICE
    cached = _CACHE.get(model_id)
    if cached:
        return "already_loaded"
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device).half()
    _CACHE.put(model_id, model, tok)
    return "loaded"


def _unload_model(model_id: str) -> str:
    _CACHE.evict(model_id)
    return "unloaded"


def _format_model_data(model_id: str, data: Any) -> Dict[str, Any]:
    """Format model data into a consistent structure.
    
    Args:
        model_id: The ID of the model
        data: The data associated with the model (typically a timestamp)
        
    Returns:
        A dictionary with structured model information
    """
    if isinstance(data, (int, float)):
        # For timestamp data (most common case)
        return {
            "load_time": data,
            "status": "loaded",
            "engine": "haystack"
        }
    elif data is None:
        # For None values
        return {
            "status": "unknown",
            "engine": "haystack",
            "data": "none"
        }
    else:
        # For any other type (strings, tuples, etc.)
        return {
            "status": "unknown",
            "engine": "haystack",
            "data": str(data)
        }

def _get_model_info() -> Dict[str, Any]:
    """Get information about all loaded models.
    
    Returns:
        Dictionary mapping model IDs to structured model information
    """
    # Get raw cache information
    cache_items = _CACHE.info()
    print(f"[ModelManager] Cache info: {cache_items}")
    
    # Process each model and format its data consistently
    result: Dict[str, Dict[str, Any]] = {}
    for model_id, data in cache_items.items():
        result[model_id] = _format_model_data(model_id, data)
    
    return result


def _debug_cache() -> Dict[str, Any]:
    """Get detailed debug information about the cache state.
    
    This is useful for troubleshooting model loading issues.
    
    Returns:
        Dictionary with detailed debug information
    """
    # Get the current state of the cache
    debug_info = {
        "cache_size": len(_CACHE._data),
        "max_cache_size": _CACHE._max,
        "keys": list(_CACHE._data.keys()),
        "debug": str(_CACHE),
    }
    print(f"[ModelManager] Debug info: {debug_info}")
    return debug_info


def _shutdown_server() -> str:
    """Shutdown the server gracefully.
    
    Returns:
        Status message
    """
    # Run this in a separate thread to avoid blocking the response
    def delayed_shutdown() -> None:
        # Give time for the response to be sent back
        time.sleep(0.5)
        # Force exit which will trigger the atexit handler
        os._exit(0)
    
    print("[ModelManager] Shutdown requested by client - server will exit")
    threading.Thread(target=delayed_shutdown, daemon=True).start()
    return "shutdown_initiated"


def _handle_request(request: Dict[str, Any]) -> Dict[str, Any]:
    """Handle a JSON-RPC request.
    
    Args:
        request: The request dictionary
        
    Returns:
        The response dictionary
    """
    if not isinstance(request, dict):
        return {"error": "Invalid request format, expected dictionary"}
    
    # Extract the action from the request
    action = request.get("action")
    if not action:
        return {"error": "Missing 'action' in request"}
    
    # Process the request based on the action
    try:
        if action == "ping":
            return {"result": "pong"}
        
        elif action == "load":
            # Load a model into memory
            model_id = request.get("model_id")
            if not model_id:
                return {"error": "Missing 'model_id' in load request"}
            
            device = request.get("device")
            load_result = _load_model(model_id, device)
            return {"result": load_result}
        
        elif action == "unload":
            # Unload a model from memory
            model_id = request.get("model_id")
            if not model_id:
                return {"error": "Missing 'model_id' in unload request"}
            
            unload_result = _unload_model(model_id)
            return {"result": unload_result}
        
        elif action == "embed":
            # Generate embeddings for texts
            model_id = request.get("model_id")
            if not model_id:
                return {"error": "Missing 'model_id' in embed request"}
            
            texts = request.get("texts")
            if not texts:
                return {"error": "Missing 'texts' in embed request"}
            
            # Import embedding function here to avoid circular imports
            from src.model_engine.engines.haystack.runtime.embedding import calculate_embeddings
            
            # Optional parameters
            pooling_strategy = request.get("pooling", "cls")
            normalize = request.get("normalize", True)
            max_length = request.get("max_length")
            
            # Generate embeddings
            result = calculate_embeddings(
                model_id=model_id,
                texts=texts,
                pooling_strategy=pooling_strategy,
                normalize=normalize,
                max_length=max_length
            )
            return {"result": result}
        
        elif action == "info":
            # Get information about loaded models
            info = _get_model_info()
            return {"result": info}
        
        elif action == "debug":
            # Get detailed debug information
            debug_info = _debug_cache()
            return {"result": debug_info}
        
        elif action == "shutdown":
            # Shutdown the server
            result = _shutdown_server()
            return {"result": result}
        
        else:
            # Unknown action
            return {"error": f"Unknown action: {action}"}
        
    except Exception as e:
        # Handle any exceptions
        print(f"[ModelManager] Error handling request: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


def _handle_conn(conn: socket.socket) -> None:
    """Handle a client connection, processing requests and sending responses.
    
    Args:
        conn: Socket connection to client
    """
    with conn:
        try:
            data = conn.recv(65536)
            
            # Process data if we received any
            if data:
                try:
                    # Parse as JSON
                    req = json.loads(data.decode())
                    
                    # Protect against non-dict input
                    if not isinstance(req, dict):
                        resp = {"error": "request must be a JSON object"}
                    else:
                        resp = _handle_request(req)
                except json.JSONDecodeError:
                    resp = {"error": "invalid JSON"}
                except Exception as e:
                    resp = {"error": f"unhandled error: {e}"}
                    
                # Send response back to client
                conn.sendall(json.dumps(resp).encode())
        except Exception as e:  # noqa: BLE001
            print(f"[ModelManager] Error handling request: {e}")
            resp = {"error": f"Error: {e}"}
            try:
                conn.sendall(json.dumps(resp).encode())
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Server mainloop
# ---------------------------------------------------------------------------

def run_server(socket_path: Optional[str] = SOCKET_PATH) -> None:
    # Handle None case with default value
    actual_socket_path = socket_path if socket_path is not None else SOCKET_PATH
    
    # Clean up stale socket
    sock_path = Path(actual_socket_path)
    if sock_path.exists():
        sock_path.unlink()

    server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        server_sock.bind(actual_socket_path)
    except Exception as e:
        print(f"[ModelManager] Failed to bind socket {actual_socket_path}: {e}", flush=True)
        raise
    os.chmod(actual_socket_path, 0o660)
    server_sock.listen()
    print(f"[ModelManager] Listening on {socket_path}", flush=True)

    def _shutdown(*_: object) -> None:
        server_sock.close()
        if sock_path.exists():
            sock_path.unlink()
        print("[ModelManager] Shutdown complete", flush=True)
        exit(0)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _shutdown)
    atexit.register(_shutdown)

    while True:
        conn, _ = server_sock.accept()
        threading.Thread(target=_handle_conn, args=(conn,), daemon=True).start()


if __name__ == "__main__":
    import sys
    try:
        sock = sys.argv[1] if len(sys.argv) > 1 else SOCKET_PATH
        run_server(sock)
    except Exception as e:
        print(f"[ModelManager] Server crashed: {e}", flush=True)
        sys.exit(1)
