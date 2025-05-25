#!/usr/bin/env python3
"""
Simplified Domain-Specific Ollama Integration Example

This script demonstrates domain-specific model selection
and automatic fallback handling with Ollama.
"""

import asyncio
import logging
import os
import requests
from typing import Dict, Any, List, Optional
import numpy as np
import ollama

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Domain identifiers
DOMAIN_CODE = "code"
DOMAIN_GENERAL = "general"

# Domain-specific model configuration
DOMAIN_MODELS = {
    DOMAIN_CODE: {
        "model": os.getenv("OLLAMA_CODE_MODEL", "tinyllama"),  # Use tinyllama as we pulled this model
        "embed_model": "tinyllama",
        "parameters": {
            "temperature": 0.2,  # Lower temperature for more precise code responses
            "top_p": 0.9,
            "top_k": 40,
            "max_tokens": 2048,  # Larger context for code generation
        }
    },
    DOMAIN_GENERAL: {
        "model": os.getenv("OLLAMA_GENERAL_MODEL", "tinyllama"),  # Use tinyllama for both now
        "embed_model": "tinyllama",
        "parameters": {
            "temperature": 0.7,  # Higher temperature for more creative responses
            "top_p": 0.9,
            "top_k": 40,
            "max_tokens": 2048,
        }
    }
}

# Code indicators for domain detection
CODE_INDICATORS = [
    ".py", ".js", ".cpp", ".java", ".go", ".rust", ".c", ".h", ".rb", 
    "function", "class", "def ", "import ", "from ", "#include", 
    "```python", "```javascript", "```cpp", "```java", "```go", "```rust",
    "package", "namespace", "module", "struct", "interface"
]

# Example content for different domains
CODE_EXAMPLE = """
def fibonacci(n):
    # Calculate the Fibonacci sequence up to n
    a, b = 0, 1
    result = []
    while a < n:
        result.append(a)
        a, b = b, a + b
    return result

# Example usage
fib_sequence = fibonacci(100)
print(f"Fibonacci sequence up to 100: {fib_sequence}")
"""

GENERAL_EXAMPLE = """
Climate change is one of the most pressing challenges of our time.
Rising global temperatures are causing more frequent extreme weather events,
sea level rise, and disruptions to ecosystems worldwide. Addressing this
challenge requires international cooperation, policy changes, and
technological innovation to reduce greenhouse gas emissions.
"""


def is_ollama_available(host: str = "http://localhost:11434") -> bool:
    """Check if Ollama service is available."""
    try:
        response = requests.get(f"{host}/api/tags", timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        logger.warning(f"Ollama service not available at {host}")
        return False


def detect_domain(content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    """Detect the domain of the content."""
    # Check metadata first if available
    if metadata and metadata.get("domain"):
        return metadata["domain"]
        
    # Check for code indicators
    if any(indicator in content for indicator in CODE_INDICATORS):
        return DOMAIN_CODE
    
    # Default to general domain
    return DOMAIN_GENERAL


def get_model_config_for_domain(domain: str) -> Dict[str, Any]:
    """Get model configuration for a specific domain."""
    # If domain not found, fall back to general
    if domain not in DOMAIN_MODELS:
        logger.warning(f"Domain '{domain}' not found, falling back to general domain")
        domain = DOMAIN_GENERAL
    
    return DOMAIN_MODELS[domain]


async def generate_text(
    prompt: str,
    content_for_detection: Optional[str] = None,
    explicit_domain: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """Generate text using domain-specific model selection."""
    # Check if Ollama is available
    if not is_ollama_available():
        return "Ollama service is not available. Please ensure it's running."
    
    # Get domain-specific configuration
    content_to_detect = content_for_detection or prompt
    
    # Determine domain
    if explicit_domain:
        detected_domain = explicit_domain
    else:
        detected_domain = detect_domain(content_to_detect, metadata)
    
    # Get model configuration
    domain_config = get_model_config_for_domain(detected_domain)
    
    # Log the selected model
    logger.info(f"Using {domain_config['model']} for domain: {detected_domain}")
    
    # Create Ollama client
    client = ollama.AsyncClient(host="http://localhost:11434")
    
    # Prepare messages
    messages = [
        {"role": "system", "content": f"You are an AI assistant specialized in {detected_domain} topics."},
        {"role": "user", "content": prompt}
    ]
    
    # Generate text with the domain-specific model
    # Ollama client expects options in a separate parameter
    response = await client.chat(
        model=domain_config["model"],
        messages=messages,
        options=domain_config.get("parameters", {})
    )
    
    return response["message"]["content"]


async def generate_embeddings(texts: List[str], domain: Optional[str] = None) -> List[List[float]]:
    """Generate embeddings for the given texts using domain-specific model."""
    # Check if Ollama is available
    if not is_ollama_available():
        logger.error("Ollama service is not available. Please ensure it's running.")
        return []
    
    # Get domain-specific configuration
    domain_config = get_model_config_for_domain(domain or DOMAIN_GENERAL)
    
    # Generate embeddings
    embed_model = domain_config.get("embed_model")
    logger.info(f"Using embedding model: {embed_model}")
    
    client = ollama.AsyncClient(host="http://localhost:11434")
    all_embeddings = []
    
    for text in texts:
        # Call the embedding function with the configured model
        # Make sure to use proper parameter names for the Ollama client
        response = await client.embeddings(model=embed_model, prompt=text)
        all_embeddings.append(response["embedding"])
    
    return all_embeddings


async def demo_domain_specific_generation():
    """Run a demonstration of domain-specific text generation."""
    # Check if Ollama is available
    if not is_ollama_available():
        logger.error("Ollama is not available. Please start the Ollama service.")
        return
    
    # 1. Code domain example
    logger.info("\n=== Code Domain Example ===")
    code_prompt = "Write a Python function to find all prime numbers up to n using the Sieve of Eratosthenes algorithm."
    code_result = await generate_text(
        prompt=code_prompt,
        content_for_detection=CODE_EXAMPLE,
        metadata={"domain": DOMAIN_CODE}
    )
    print(f"\nPrompt: {code_prompt}\n")
    print(f"Generated Code:\n{code_result}\n")
    
    # 2. General domain example
    logger.info("\n=== General Domain Example ===")
    general_prompt = "Explain the concept of ocean acidification and its impact on marine ecosystems."
    general_result = await generate_text(
        prompt=general_prompt,
        content_for_detection=GENERAL_EXAMPLE
    )
    print(f"\nPrompt: {general_prompt}\n")
    print(f"Generated Text:\n{general_result}\n")
    
    # 3. Domain detection example
    logger.info("\n=== Domain Detection Example ===")
    mixed_prompt = "How would you implement a binary search tree in Python?"
    mixed_result = await generate_text(
        prompt=mixed_prompt
    )
    print(f"\nPrompt: {mixed_prompt}\n")
    print(f"Detected Domain and Generated Text:\n{mixed_result}\n")
    
    # 4. Embedding example
    logger.info("\n=== Embedding Example ===")
    embed_texts = [
        "This is a sample text for embedding generation.",
        "def example_function(): return 'Hello, World!'"
    ]
    embeddings = await generate_embeddings(embed_texts)
    print(f"Generated {len(embeddings)} embeddings of dimension {len(embeddings[0])}")


if __name__ == "__main__":
    asyncio.run(demo_domain_specific_generation())
