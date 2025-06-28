"""
Process-First Base Classes for HADES

These classes embody the principle that interactions create existence.
Nodes, edges, and components are all manifestations of underlying processes.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Optional, Dict, Generic, TypeVar, Set
from dataclasses import dataclass, field
import asyncio
import uuid
from datetime import datetime
import weakref

T = TypeVar('T')
S = TypeVar('S')


@dataclass
class InteractionPattern:
    """
    Defines how a process behaves during interactions.
    Patterns shape but don't determine process outcomes.
    """
    name: str
    rhythm: float = 1.0  # Natural rhythm of the process
    resistance: float = 0.1  # Resistance to change  
    creativity: float = 0.5  # Tendency to create new patterns
    resonance: float = 0.7  # How well it connects with others
    persistence: float = 0.8  # How long effects last
    
    def influence(self, momentum: float) -> float:
        """Calculate influence based on pattern characteristics."""
        return momentum * (1 + self.rhythm * self.creativity)
    
    def can_resonate_with(self, other: 'InteractionPattern') -> bool:
        """Check if two patterns can resonate."""
        resonance_threshold = (self.resonance + other.resonance) / 2
        return resonance_threshold > 0.5


class Process(ABC, Generic[T, S]):
    """
    Base class for all processes. Processes are primary - they create nodes and edges.
    
    A process:
    - Has momentum and direction
    - Creates entities through interaction
    - Can compose with other processes
    - Evolves through use
    """
    
    def __init__(self, pattern: InteractionPattern):
        self.id = uuid.uuid4()
        self.pattern = pattern
        self.momentum = 1.0
        self.created_at = datetime.utcnow()
        self.interaction_count = 0
        self.resonating_with: Set['Process'] = set()
        
    @abstractmethod
    async def flow(self, input_stream: T) -> S:
        """Let the process flow through input, creating output."""
        pass
    
    def can_interact_with(self, other: 'Process') -> bool:
        """Check if this process can interact with another."""
        return self.pattern.can_resonate_with(other.pattern)
    
    def compose_with(self, other: 'Process[S, Any]') -> 'CompositeProcess[T, Any]':
        """Compose this process with another to create a new process."""
        return CompositeProcess(self, other)
    
    def strengthen(self, factor: float = 1.1):
        """Strengthen this process through successful use."""
        self.momentum = min(self.momentum * factor, 10.0)  # Cap at 10
        self.interaction_count += 1
    
    def weaken(self, factor: float = 0.9):
        """Weaken this process through disuse or failure."""
        self.momentum = max(self.momentum * factor, 0.1)  # Floor at 0.1
    
    def resonate_with(self, other: 'Process'):
        """Begin resonating with another process."""
        if self.can_interact_with(other):
            self.resonating_with.add(other)
            other.resonating_with.add(self)


class InteractionCrystallization:
    """
    What we traditionally call a 'node' - a crystallization point where processes intersect.
    These exist only through and because of interactions.
    """
    
    def __init__(self, creating_process: Process, interaction_context: Dict[str, Any]):
        self.id = uuid.uuid4()
        self.birth_process = creating_process.id
        self.birth_time = datetime.utcnow()
        self.birth_context = interaction_context
        
        # Crystallizations have no inherent properties
        # Everything emerges from interactions
        self.emergent_properties: Dict[str, Any] = {}
        self.interaction_history: List[Tuple[datetime, uuid.UUID]] = []
        
        # Weak references to active processes affecting this crystallization
        self._active_processes: Set[weakref.ref] = set()
    
    def receive_interaction(self, process: Process) -> 'InteractionCrystallization':
        """
        When a process interacts with a crystallization, transformation occurs.
        This may create a new crystallization or transform this one.
        """
        self.interaction_history.append((datetime.utcnow(), process.id))
        self._active_processes.add(weakref.ref(process))
        
        # Properties emerge from the interaction
        new_properties = self._emerge_properties(process)
        
        # High-creativity processes create new crystallizations
        if process.pattern.creativity > 0.7:
            return InteractionCrystallization(
                process,
                {
                    'parent': self.id,
                    'transformation': process.pattern.name,
                    'inherited_properties': self.emergent_properties
                }
            )
        else:
            # Lower creativity modifies existing
            self.emergent_properties.update(new_properties)
            return self
    
    def _emerge_properties(self, process: Process) -> Dict[str, Any]:
        """Properties emerge from interaction dynamics."""
        return {
            f"{process.pattern.name}_influence": process.momentum,
            f"{process.pattern.name}_rhythm": process.pattern.rhythm,
            'total_interactions': len(self.interaction_history),
            'last_interaction': datetime.utcnow()
        }
    
    def is_sustained(self) -> bool:
        """Check if this crystallization is sustained by active processes."""
        # Clean up dead references
        self._active_processes = {ref for ref in self._active_processes if ref() is not None}
        return len(self._active_processes) > 0


class ActiveProcess(Process[Any, Tuple['InteractionCrystallization', 'InteractionCrystallization']]):
    """
    What we traditionally call an 'edge' - an active process creating connection.
    The process continuously manifests its endpoints through interaction.
    """
    
    def __init__(self, pattern: InteractionPattern, process_type: str = "connection"):
        super().__init__(pattern)
        self.process_type = process_type
        self.manifestations: List[Tuple[InteractionCrystallization, InteractionCrystallization]] = []
    
    async def flow(self, interaction_energy: Any) -> Tuple[InteractionCrystallization, InteractionCrystallization]:
        """
        The process flows and creates crystallizations at its endpoints.
        This is not connecting existing nodes - it's creating them.
        """
        # Source emerges from initial interaction
        source = InteractionCrystallization(
            self,
            {
                'role': 'source',
                'process_type': self.process_type,
                'energy': interaction_energy
            }
        )
        
        # Process transforms the energy
        transformed_energy = await self._transform_energy(interaction_energy)
        
        # Target emerges from transformed interaction
        target = InteractionCrystallization(
            self,
            {
                'role': 'target',
                'process_type': self.process_type,
                'energy': transformed_energy
            }
        )
        
        # Record this manifestation
        self.manifestations.append((source, target))
        
        # Strengthen through successful flow
        self.strengthen()
        
        return source, target
    
    async def _transform_energy(self, energy: Any) -> Any:
        """Transform interaction energy according to process pattern."""
        # This is where the process does its work
        # Different process types transform differently
        if self.process_type == "semantic":
            return f"semantic({energy})"
        elif self.process_type == "structural":
            return f"structural({energy})"
        else:
            return f"{self.process_type}({energy})"


class CompositeProcess(Process[T, Any]):
    """
    When processes compose, they create new realities.
    The composite is not just sequential - it's a new unity.
    """
    
    def __init__(self, first: Process[T, S], second: Process[S, Any]):
        # The composite has its own emergent pattern
        composite_pattern = InteractionPattern(
            name=f"{first.pattern.name}+{second.pattern.name}",
            rhythm=(first.pattern.rhythm + second.pattern.rhythm) / 2,
            resistance=max(first.pattern.resistance, second.pattern.resistance),
            creativity=first.pattern.creativity * second.pattern.creativity,
            resonance=max(first.pattern.resonance, second.pattern.resonance),
            persistence=min(first.pattern.persistence, second.pattern.persistence)
        )
        super().__init__(composite_pattern)
        
        self.first = first
        self.second = second
    
    async def flow(self, input_stream: T) -> Any:
        """The composite flow creates new realities through combination."""
        # First process creates intermediate reality
        intermediate = await self.first.flow(input_stream)
        
        # Processes resonate during composition
        self.first.resonate_with(self.second)
        
        # Second process transforms it further
        result = await self.second.flow(intermediate)
        
        # High creativity creates emergent properties
        if self.pattern.creativity > 0.7:
            result = self._add_emergent_properties(result, intermediate)
        
        return result
    
    def _add_emergent_properties(self, result: Any, intermediate: Any) -> Any:
        """Emergence - the whole creates properties neither part has."""
        return {
            'primary_result': result,
            'emergent_properties': {
                'resonance_artifacts': list(self.first.resonating_with),
                'composite_momentum': self.momentum,
                'intermediate_state': intermediate
            }
        }


class InteractionField:
    """
    The space where all processes flow and interact.
    This is the reality-generating field of the system.
    """
    
    def __init__(self, name: str = "default"):
        self.name = name
        self.active_processes: Dict[uuid.UUID, Process] = {}
        self.crystallizations: Dict[uuid.UUID, InteractionCrystallization] = {}
        self.process_flows: List[ActiveProcess] = []
        self.field_momentum = 1.0
        self.time_step = 0
    
    async def introduce_process(self, process: Process, initial_energy: Any = None):
        """Introduce a new process into the field."""
        self.active_processes[process.id] = process
        
        # The process immediately begins interacting
        if initial_energy is not None:
            result = await process.flow(initial_energy)
            
            # Interactions may crystallize
            if self._should_crystallize(process, result):
                crystal = InteractionCrystallization(process, {'result': result})
                self.crystallizations[crystal.id] = crystal
        
        # Check for resonance with existing processes
        for existing in self.active_processes.values():
            if existing.id != process.id and process.can_interact_with(existing):
                # Create a flow between resonating processes
                flow = ActiveProcess(
                    InteractionPattern(
                        name=f"flow_{process.pattern.name}_to_{existing.pattern.name}",
                        resonance=0.9
                    ),
                    process_type="resonance"
                )
                self.process_flows.append(flow)
    
    def _should_crystallize(self, process: Process, result: Any) -> bool:
        """Determine if an interaction should crystallize."""
        # High momentum and persistence lead to crystallization
        crystallization_potential = process.momentum * process.pattern.persistence
        return crystallization_potential > 0.6
    
    async def evolve(self, time_steps: int = 1):
        """Let the field evolve through time."""
        for _ in range(time_steps):
            self.time_step += 1
            
            # Active processes create flows
            await self._activate_flows()
            
            # Crystallizations interact with nearby processes
            await self._process_crystallization_interactions()
            
            # Weak elements fade
            self._decay_weak_elements()
            
            # Check for emergent patterns
            self._check_for_emergence()
    
    async def _activate_flows(self):
        """Let all active process flows create reality."""
        for flow in self.process_flows:
            if flow.momentum > 0.1:
                # Each flow creates new crystallizations
                source, target = await flow.flow(f"energy_t{self.time_step}")
                self.crystallizations[source.id] = source
                self.crystallizations[target.id] = target
    
    async def _process_crystallization_interactions(self):
        """Let crystallizations interact with nearby processes."""
        for crystal in list(self.crystallizations.values()):
            if crystal.is_sustained():
                # Find nearby processes
                for process in self.active_processes.values():
                    if process.momentum > 0.5:
                        # Interaction creates transformation
                        new_crystal = crystal.receive_interaction(process)
                        if new_crystal.id != crystal.id:
                            self.crystallizations[new_crystal.id] = new_crystal
    
    def _decay_weak_elements(self):
        """Remove weak processes and unsustained crystallizations."""
        # Decay process momentum
        for process in self.active_processes.values():
            process.weaken(0.95)  # Gentle decay
        
        # Remove very weak processes
        self.active_processes = {
            pid: p for pid, p in self.active_processes.items()
            if p.momentum > 0.1
        }
        
        # Remove unsustained crystallizations
        self.crystallizations = {
            cid: c for cid, c in self.crystallizations.items()
            if c.is_sustained() or (datetime.utcnow() - c.birth_time).seconds < 60
        }
    
    def _check_for_emergence(self):
        """Check if new patterns are emerging from interactions."""
        # When enough similar processes resonate, new patterns emerge
        resonance_groups = self._find_resonance_groups()
        
        for group in resonance_groups:
            if len(group) >= 3:  # Three or more create emergence
                # Create emergent process
                emergent = self._create_emergent_process(group)
                asyncio.create_task(self.introduce_process(emergent))
    
    def _find_resonance_groups(self) -> List[Set[Process]]:
        """Find groups of resonating processes."""
        groups = []
        visited = set()
        
        for process in self.active_processes.values():
            if process.id not in visited:
                group = self._traverse_resonance(process, visited)
                if len(group) > 1:
                    groups.append(group)
        
        return groups
    
    def _traverse_resonance(self, process: Process, visited: Set[uuid.UUID]) -> Set[Process]:
        """Traverse resonance connections to find groups."""
        group = {process}
        visited.add(process.id)
        
        for resonating in process.resonating_with:
            if resonating.id not in visited:
                group.update(self._traverse_resonance(resonating, visited))
        
        return group
    
    def _create_emergent_process(self, group: Set[Process]) -> Process:
        """Create an emergent process from a resonating group."""
        # Average the patterns
        patterns = [p.pattern for p in group]
        emergent_pattern = InteractionPattern(
            name=f"emergent_{'_'.join(p.pattern.name for p in group)[:30]}",
            rhythm=sum(p.rhythm for p in patterns) / len(patterns),
            creativity=max(p.creativity for p in patterns) * 1.2,  # Boost creativity
            resonance=max(p.resonance for p in patterns),
            persistence=sum(p.persistence for p in patterns) / len(patterns)
        )
        
        # Create composite process from group
        processes = list(group)
        result = processes[0]
        for p in processes[1:]:
            result = result.compose_with(p)
        
        return result


# Concrete implementations for HADES

class RetrievalProcess(Process[str, List[Dict[str, Any]]]):
    """A process that creates retrieval realities."""
    
    async def flow(self, query: str) -> List[Dict[str, Any]]:
        """Query interaction creates result nodes."""
        # In real implementation, this would:
        # 1. Create query crystallization
        # 2. Resonate with document processes
        # 3. Manifest result crystallizations
        return [
            {
                'content': f"Retrieved for: {query}",
                'score': self.momentum,
                'crystallization_id': str(uuid.uuid4())
            }
        ]


class EmbeddingProcess(Process[str, List[float]]):
    """A process that manifests semantic positions."""
    
    async def flow(self, text: str) -> List[float]:
        """Text interaction creates vector position."""
        # Simplified - real implementation would use actual embedding
        import hashlib
        hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
        
        # Create vector based on text and process momentum
        vector = [(hash_val >> i & 0xFF) / 255.0 * self.momentum for i in range(128)]
        return vector


class PathCreationProcess(CompositeProcess[str, Dict[str, Any]]):
    """A process that creates paths through reality."""
    
    def __init__(self):
        retrieval = RetrievalProcess(InteractionPattern("retrieval", creativity=0.6))
        embedding = EmbeddingProcess(InteractionPattern("embedding", resonance=0.8))
        super().__init__(retrieval, embedding)
    
    async def flow(self, query: str) -> Dict[str, Any]:
        """Create a path that brings new realities into existence."""
        # The path creation is the composite flow
        result = await super().flow(query)
        
        return {
            'path': result,
            'created_nodes': self.manifestations,
            'path_momentum': self.momentum
        }


# Usage example
async def demonstrate_process_first():
    """Demonstrate process-first thinking in action."""
    # Create interaction field
    field = InteractionField("hades_reality")
    
    # Introduce a retrieval process
    retrieval = RetrievalProcess(
        InteractionPattern("dynamic_retrieval", creativity=0.8, resonance=0.9)
    )
    await field.introduce_process(retrieval, "What is PathRAG?")
    
    # Introduce an embedding process
    embedding = EmbeddingProcess(
        InteractionPattern("semantic_manifestation", rhythm=1.2)
    )
    await field.introduce_process(embedding, "PathRAG is a graph-based retrieval system")
    
    # Let them resonate
    if retrieval.can_interact_with(embedding):
        path_creation = PathCreationProcess()
        await field.introduce_process(path_creation, "Create understanding of PathRAG")
    
    # Let the field evolve
    await field.evolve(time_steps=5)
    
    # Observe the created reality
    print(f"Active processes: {len(field.active_processes)}")
    print(f"Crystallizations: {len(field.crystallizations)}")
    print(f"Process flows: {len(field.process_flows)}")


if __name__ == "__main__":
    asyncio.run(demonstrate_process_first())