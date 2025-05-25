#!/usr/bin/env python3
"""
Domain-Specific Ollama Integration Example for HADES-PathRAG

This script demonstrates how to use domain-specific model selection
and automatic fallback handling with Ollama in HADES-PathRAG.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import PathRAG components
try:
    # Try importing from src directory
    from src.config.llm_config import (
        get_domain_specific_config,
        is_ollama_available,
        with_fallback_provider,
        DOMAIN_CODE,
        DOMAIN_GENERAL
    )
    from src.pathrag.llm import ollama_model_complete, ollama_embed
except ImportError:
    # Fall back to installed package
    from PathRAG.config.llm_config import (
        get_domain_specific_config,
        is_ollama_available,
        with_fallback_provider,
        DOMAIN_CODE,
        DOMAIN_GENERAL
    )
    from PathRAG.llm import ollama_model_complete, ollama_embed


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


@with_fallback_provider(fallback_provider="openai")
async def generate_text_with_domain_detection(
    prompt: str,
    content_for_detection: Optional[str] = None,
    explicit_domain: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generate text using domain-specific model selection.
    
    Args:
        prompt: The prompt for text generation
        content_for_detection: Content to use for domain detection
        explicit_domain: Explicitly specified domain (overrides detection)
        metadata: Additional metadata
        
    Returns:
        Generated text
    """
    # Get domain-specific configuration
    content_to_detect = content_for_detection or prompt
    domain_config = get_domain_specific_config(
        content=content_to_detect,
        domain=explicit_domain,
        metadata=metadata
    )
    
    # Log the selected model
    detected_domain = explicit_domain or (
        "code" if "code" in domain_config.get("model", "").lower() else "general"
    )
    logger.info(f"Using {domain_config['model']} for domain: {detected_domain}")
    
    # Create a mock hashing_kv object with the selected model
    class HashingKV:
        def __init__(self, model_name):
            self.global_config = {"llm_model_name": model_name}
    
    # Generate text with the domain-specific model
    result = await ollama_model_complete(
        prompt=prompt,
        system_prompt=f"You are an AI assistant specialized in {detected_domain} topics.",
        hashing_kv=HashingKV(domain_config["model"]),
        host=domain_config.get("host"),
        timeout=domain_config.get("timeout", 60),
        **domain_config.get("parameters", {})
    )
    
    return result


async def generate_embeddings(texts: List[str], domain: Optional[str] = None) -> List[List[float]]:
    """
    Generate embeddings for the given texts using domain-specific model.
    
    Args:
        texts: List of texts to generate embeddings for
        domain: Optional domain to use for model selection
        
    Returns:
        List of embeddings
    """
    # Get domain-specific configuration
    domain_config = get_domain_specific_config(domain=domain)
    
    # Generate embeddings
    embed_model = domain_config.get("embed_model")
    logger.info(f"Using embedding model: {embed_model}")
    
    all_embeddings = []
    for text in texts:
        # Call the embedding function directly with the configured model
        embeddings = await ollama_embed([text], embed_model=embed_model)
        all_embeddings.append(embeddings.tolist()[0])
    
    return all_embeddings


async def demo_domain_specific_generation():
    """Run a demonstration of domain-specific text generation."""
    # Check if Ollama is available
    if not is_ollama_available():
        logger.warning("Ollama is not available. The script will use fallback mechanisms.")
    
    # 1. Code domain example
    logger.info("\n=== Code Domain Example ===")
    code_prompt = "Write a Python function to find all prime numbers up to n using the Sieve of Eratosthenes algorithm."
    code_result = await generate_text_with_domain_detection(
        prompt=code_prompt,
        content_for_detection=CODE_EXAMPLE,
        metadata={"domain": DOMAIN_CODE}
    )
    print(f"\nPrompt: {code_prompt}\n")
    print(f"Generated Code:\n{code_result}\n")
    
    # 2. General domain example
    logger.info("\n=== General Domain Example ===")
    general_prompt = "Explain the concept of ocean acidification and its impact on marine ecosystems."
    general_result = await generate_text_with_domain_detection(
        prompt=general_prompt,
        content_for_detection=GENERAL_EXAMPLE
    )
    print(f"\nPrompt: {general_prompt}\n")
    print(f"Generated Text:\n{general_result}\n")
    
    # 3. Domain detection example
    logger.info("\n=== Domain Detection Example ===")
    mixed_prompt = "How would you implement a binary search tree in Python?"
    mixed_result = await generate_text_with_domain_detection(
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
