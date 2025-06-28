#!/usr/bin/env python3
"""
Demo: Process Ethnography with Existing Pipeline
Experience data transformation from the data's perspective.
"""

import asyncio
import sys
from pathlib import Path
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.pipelines.process_ethnography import DataJourney, DataBecoming

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def demo_data_journey():
    """Experience the journey of research paper becoming knowledge."""
    
    # Sample text from one of our research papers
    research_text = """
    SACL: Semantic-Augmented Code Retrieval combats the textual bias problem
    in code search by incorporating both textual and semantic information.
    Traditional approaches suffer when natural language queries don't match
    code syntax. SACL addresses this through dual-channel processing.
    """
    
    logger.info("=== Beginning Data Ethnography Demo ===")
    logger.info("We will experience transformation from the data's perspective")
    logger.info("")
    
    # Create journey orchestrator
    journey = DataJourney()
    
    # Journey with the data
    logger.info("Starting journey with research paper text...")
    final_form = await journey.journey_with_data(research_text)
    
    # Visualize the complete journey
    logger.info("")
    print(journey.visualize_journey(final_form))
    
    # Show the data's memories
    logger.info("\n=== Data's Autobiographical Memory ===")
    memory = final_form.memory
    logger.info(f"Birth moment: {memory.birth_moment}")
    logger.info(f"Transformations experienced: {len(memory.transformations)}")
    logger.info(f"Relationships formed: {len(memory.relationships_formed)}")
    logger.info(f"Spaces inhabited: {memory.spaces_inhabited}")
    
    # Philosophical reflection
    logger.info("\n=== Philosophical Insights ===")
    logger.info("- The data didn't get 'processed' - it BECAME through interaction")
    logger.info("- Each transformation was a co-creation between data and process")
    logger.info("- Relationships emerged naturally from resonance patterns")
    logger.info("- The graph form is not storage but a living network")
    
    return final_form


async def demo_multiple_data_interactions():
    """Show how multiple data pieces interact and resonate."""
    
    texts = [
        "PathRAG enables multi-hop reasoning through knowledge graphs",
        "Process-first architecture treats interactions as primary",
        "ISNE creates graph-aware embeddings through node relationships"
    ]
    
    logger.info("\n=== Multiple Data Streams Demo ===")
    
    journey = DataJourney()
    becomings = []
    
    for text in texts:
        logger.info(f"\nProcessing: {text[:50]}...")
        becoming = await journey.journey_with_data(text)
        becomings.append(becoming)
    
    # Show how they might resonate
    logger.info("\n=== Potential Resonances ===")
    logger.info("- PathRAG and ISNE both work with graphs - natural resonance")
    logger.info("- Process-first validates their interaction patterns")
    logger.info("- Together they form an emergent understanding")
    
    return becomings


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Demo process ethnography - experiencing data transformation"
    )
    parser.add_argument("--multiple", action="store_true",
                       help="Demo with multiple interacting data streams")
    
    args = parser.parse_args()
    
    try:
        if args.multiple:
            asyncio.run(demo_multiple_data_interactions())
        else:
            asyncio.run(demo_data_journey())
            
        logger.info("\n✅ Process ethnography demo completed!")
        logger.info("This shows data transformation from within, not from above.")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise