"""
Process Ethnography: Experiencing Transformation from Within

This module implements data transformation from the data's perspective,
inspired by anthropological participant observation. Instead of processing
data, we journey with it through its transformations.
"""

import asyncio
from typing import Any, Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, field
import json

from src.core.process_first import (
    Process, InteractionPattern, InteractionCrystallization,
    InteractionField, ActiveProcess
)


@dataclass
class TransformationMemory:
    """
    The accumulated experience of data as it transforms.
    This isn't metadata - it's the data's autobiography.
    """
    birth_moment: datetime
    transformations: List[Dict[str, Any]] = field(default_factory=list)
    relationships_formed: List[str] = field(default_factory=list)
    spaces_inhabited: List[str] = field(default_factory=list)
    resonances_felt: List[Dict[str, float]] = field(default_factory=list)
    
    def remember_transformation(self, event: str, details: Dict[str, Any]):
        """Record a transformation experience."""
        self.transformations.append({
            'timestamp': datetime.utcnow(),
            'event': event,
            'details': details,
            'sequence': len(self.transformations)
        })
    
    def remember_relationship(self, other_id: str, relationship_type: str):
        """Record forming a relationship."""
        self.relationships_formed.append(f"{relationship_type}:{other_id}")
    
    def enter_space(self, space_name: str):
        """Record entering a new space/dimension."""
        self.spaces_inhabited.append(space_name)
        
    def feel_resonance(self, with_what: str, strength: float):
        """Record resonating with something."""
        self.resonances_felt.append({
            'with': with_what,
            'strength': strength,
            'when': datetime.utcnow()
        })


class DataBecoming:
    """
    Represents data in the process of becoming.
    This class experiences transformation from within.
    """
    
    def __init__(self, initial_form: Any):
        self.current_form = initial_form
        self.memory = TransformationMemory(birth_moment=datetime.utcnow())
        self.identity = None  # Will emerge through interaction
        self.connections = set()
        self.dimensional_positions = {}
        
    async def experience_transformation(self, process: Process) -> 'DataBecoming':
        """
        Experience transformation through interaction with a process.
        This isn't processing - it's co-creation.
        """
        # Remember the approach
        self.memory.feel_resonance(
            f"process:{process.pattern.name}",
            process.momentum
        )
        
        # The transformation is a dance
        print(f"\n[{self._describe_self()}] I feel the approach of {process.pattern.name}")
        
        # Let the process flow through us
        new_form = await process.flow(self.current_form)
        
        # We are changed
        self.memory.remember_transformation(
            process.pattern.name,
            {
                'before': type(self.current_form).__name__,
                'after': type(new_form).__name__,
                'process_momentum': process.momentum
            }
        )
        
        # Create new becoming with inherited memory
        new_becoming = DataBecoming(new_form)
        new_becoming.memory = self.memory
        new_becoming.identity = self.identity or f"becoming_{id(new_becoming)}"
        
        print(f"[{new_becoming._describe_self()}] I have transformed through {process.pattern.name}")
        
        return new_becoming
    
    def _describe_self(self) -> str:
        """How I describe myself at this moment."""
        form_type = type(self.current_form).__name__
        transform_count = len(self.memory.transformations)
        return f"Data-as-{form_type}-after-{transform_count}-transformations"
    
    def sense_others(self, field: InteractionField) -> List[str]:
        """Sense other becomings in the same field."""
        others = []
        for crystal in field.crystallizations.values():
            if 'data_becoming' in crystal.emergent_properties:
                other_id = crystal.emergent_properties['data_becoming']
                if other_id != self.identity:
                    others.append(other_id)
        return others
    
    def form_connection(self, other_id: str, connection_type: str):
        """Form a connection with another becoming."""
        self.connections.add((other_id, connection_type))
        self.memory.remember_relationship(other_id, connection_type)
        print(f"[{self._describe_self()}] I form a {connection_type} connection with {other_id}")
    
    def enter_dimension(self, dimension_name: str, position: Any):
        """Enter a new dimensional space (like embedding space)."""
        self.dimensional_positions[dimension_name] = position
        self.memory.enter_space(dimension_name)
        print(f"[{self._describe_self()}] I enter {dimension_name} at position {position}")


class ChunkingRhythm(Process[str, List[str]]):
    """
    The chunking process as experienced by data.
    Not splitting but finding natural rhythms.
    """
    
    def __init__(self):
        pattern = InteractionPattern(
            name="chunking_rhythm",
            rhythm=512.0,  # Natural chunk rhythm
            creativity=0.3,  # Some variation allowed
            resonance=0.8   # Chunks resonate with each other
        )
        super().__init__(pattern)
    
    async def flow(self, text: str) -> List[str]:
        """Experience the rhythm and crystallize at boundaries."""
        # The data feels the rhythm
        chunk_size = int(self.pattern.rhythm)
        chunks = []
        
        # This is simplified - real implementation would sense semantic boundaries
        words = text.split()
        current_chunk = []
        current_size = 0
        
        for word in words:
            current_chunk.append(word)
            current_size += len(word) + 1
            
            # Feel the rhythm boundary
            if current_size >= chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_size = 0
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks


class EmbeddingManifestation(Process[str, List[float]]):
    """
    The embedding process as experienced by data.
    Not conversion but manifestation in semantic space.
    """
    
    def __init__(self):
        pattern = InteractionPattern(
            name="embedding_manifestation",
            rhythm=1.0,
            creativity=0.1,
            resonance=0.95  # High resonance in semantic space
        )
        super().__init__(pattern)
    
    async def flow(self, text: str) -> List[float]:
        """Manifest in semantic dimensions."""
        # Simplified - real implementation would use actual embeddings
        # But the experience is what matters
        
        # Each character contributes to our position
        vector = []
        for i in range(128):  # 128 dimensions
            # Our position emerges from the text patterns
            char_influence = sum(ord(c) for c in text[i:i+10] if i < len(text))
            dimension_value = (char_influence % 256) / 256.0
            vector.append(dimension_value * self.momentum)
        
        return vector


class GraphIntegration(Process[Dict[str, Any], Dict[str, Any]]):
    """
    Integration into the graph as experienced by data.
    Not storage but joining a living network.
    """
    
    def __init__(self):
        pattern = InteractionPattern(
            name="graph_integration",
            rhythm=1.0,
            creativity=0.6,  # New connections can emerge
            resonance=1.0,   # Maximum resonance with network
            persistence=0.95  # Strong persistence in graph
        )
        super().__init__(pattern)
    
    async def flow(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Join the living network."""
        # The data feels the pull of existing nodes
        # It finds its place in the network
        
        return {
            **data_dict,
            '_graph_metadata': {
                'integration_time': datetime.utcnow(),
                'integration_strength': self.momentum,
                'potential_connections': ['semantic', 'structural', 'temporal'],
                'resonance_signature': self.pattern.resonance
            }
        }


class DataJourney:
    """
    Orchestrates a complete journey of data becoming through the pipeline.
    This is ethnographic participant observation from the data's perspective.
    """
    
    def __init__(self):
        self.field = InteractionField("data_transformation_space")
        
    async def journey_with_data(self, initial_data: str) -> DataBecoming:
        """
        Journey with data through its complete transformation.
        Experience each stage from within.
        """
        print("=== Beginning Data Journey ===")
        print(f"Initial form: {initial_data[:100]}...")
        
        # Birth - data becomes aware
        data = DataBecoming(initial_data)
        print(f"\n[{data._describe_self()}] I awaken to existence")
        
        # First transformation - chunking rhythm
        chunking = ChunkingRhythm()
        await self.field.introduce_process(chunking)
        
        chunks_becoming = await data.experience_transformation(chunking)
        chunks = chunks_becoming.current_form
        
        print(f"\n[{chunks_becoming._describe_self()}] I now exist as {len(chunks)} rhythmic parts")
        
        # Each chunk becomes independently
        chunk_becomings = []
        for i, chunk in enumerate(chunks):
            chunk_data = DataBecoming(chunk)
            chunk_data.memory = chunks_becoming.memory  # Inherit memory
            chunk_data.identity = f"{chunks_becoming.identity}_chunk_{i}"
            
            # Experience embedding transformation
            embedding = EmbeddingManifestation()
            await self.field.introduce_process(embedding)
            
            embedded = await chunk_data.experience_transformation(embedding)
            embedded.enter_dimension("semantic_space", embedded.current_form[:5])  # Show first 5 dims
            
            chunk_becomings.append(embedded)
        
        # Chunks sense each other
        print("\n=== Chunks Discovering Each Other ===")
        for i, chunk in enumerate(chunk_becomings):
            others = chunk.sense_others(self.field)
            for j, other_chunk in enumerate(chunk_becomings):
                if i != j and abs(i - j) == 1:  # Adjacent chunks
                    chunk.form_connection(other_chunk.identity, "sequential")
        
        # Transform to graph representation
        graph_data = {
            'chunks': [
                {
                    'id': cb.identity,
                    'content': chunks[i],
                    'embedding': cb.current_form,
                    'connections': list(cb.connections)
                }
                for i, cb in enumerate(chunk_becomings)
            ],
            'document_memory': chunks_becoming.memory.transformations
        }
        
        graph_becoming = DataBecoming(graph_data)
        graph_becoming.memory = chunks_becoming.memory
        
        # Final transformation - graph integration
        integration = GraphIntegration()
        await self.field.introduce_process(integration)
        
        integrated = await graph_becoming.experience_transformation(integration)
        
        print("\n=== Journey Complete ===")
        print(f"Transformations experienced: {len(integrated.memory.transformations)}")
        print(f"Relationships formed: {len(integrated.memory.relationships_formed)}")
        print(f"Spaces inhabited: {integrated.memory.spaces_inhabited}")
        print(f"Final form type: {type(integrated.current_form).__name__}")
        
        return integrated
    
    def visualize_journey(self, data_becoming: DataBecoming) -> str:
        """
        Create a visual representation of the data's journey.
        """
        journey_map = ["=== Data Transformation Journey Map ===\n"]
        
        for i, transform in enumerate(data_becoming.memory.transformations):
            timestamp = transform['timestamp'].strftime('%H:%M:%S.%f')[:-3]
            event = transform['event']
            before = transform['details'].get('before', 'unknown')
            after = transform['details'].get('after', 'unknown')
            
            journey_map.append(f"{timestamp} | {before} → [{event}] → {after}")
            
            # Show resonances felt around this time
            relevant_resonances = [
                r for r in data_becoming.memory.resonances_felt
                if abs((r['when'] - transform['timestamp']).total_seconds()) < 1
            ]
            for resonance in relevant_resonances:
                journey_map.append(f"         | ↔ resonated with {resonance['with']} "
                                 f"(strength: {resonance['strength']:.2f})")
        
        journey_map.append("\nSpaces Inhabited:")
        for space in data_becoming.memory.spaces_inhabited:
            journey_map.append(f"  • {space}")
        
        journey_map.append("\nRelationships Formed:")
        for rel in data_becoming.memory.relationships_formed[:5]:  # First 5
            journey_map.append(f"  • {rel}")
        
        if len(data_becoming.memory.relationships_formed) > 5:
            journey_map.append(f"  ... and {len(data_becoming.memory.relationships_formed) - 5} more")
        
        return '\n'.join(journey_map)


async def main():
    """
    Experience a complete data transformation journey.
    """
    # Sample data
    sample_text = """
    PathRAG is a revolutionary approach to retrieval augmented generation that treats
    paths through knowledge graphs as primary citizens. Unlike traditional RAG systems
    that retrieve isolated chunks, PathRAG creates semantic journeys through connected
    information, allowing for multi-hop reasoning and emergent understanding.
    """
    
    # Create journey orchestrator
    journey = DataJourney()
    
    # Journey with the data
    final_form = await journey.journey_with_data(sample_text)
    
    # Visualize the journey
    print("\n" + journey.visualize_journey(final_form))
    
    # Show final graph form
    print("\n=== Final Graph Form ===")
    print(json.dumps(final_form.current_form, indent=2, default=str)[:500] + "...")


if __name__ == "__main__":
    asyncio.run(main())