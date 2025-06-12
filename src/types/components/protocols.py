"""
Component Protocol Interfaces

This module defines the protocol interfaces that all component implementations
must follow. These protocols enforce the contract compliance and enable
type-safe component swapping.

All component implementations must implement the appropriate protocol to ensure
they can be used interchangeably in the pipeline.
"""

from typing import Protocol, runtime_checkable, Dict, Any, List, Optional, TYPE_CHECKING
from abc import abstractmethod

if TYPE_CHECKING:
    pass

from .contracts import (
    DocumentProcessingInput,
    DocumentProcessingOutput,
    ChunkingInput,
    ChunkingOutput,
    EmbeddingInput,
    EmbeddingOutput,
    GraphEnhancementInput,
    GraphEnhancementOutput,
    StorageInput,
    StorageOutput,
    QueryInput,
    QueryOutput,
    ModelEngineInput,
    ModelEngineOutput,
    ComponentType
)

# ===== Base Component Protocol =====

@runtime_checkable
class BaseComponent(Protocol):
    """
    Base protocol for all HADES components.
    
    All component implementations must implement this interface to be compatible
    with the config-driven architecture.
    """
    
    @property
    def name(self) -> str:
        """Component name for identification."""
        ...
    
    @property
    def version(self) -> str:
        """Component version string."""
        ...
    
    @property
    def component_type(self) -> ComponentType:
        """Type of component."""
        ...
    
    def configure(self, config: Dict[str, Any]) -> None:
        """
        Configure component with parameters.
        
        Args:
            config: Configuration dictionary containing component parameters
            
        Raises:
            ValueError: If configuration is invalid
        """
        ...
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate configuration parameters.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            True if configuration is valid, False otherwise
        """
        ...
    
    def get_config_schema(self) -> Dict[str, Any]:
        """
        Get the JSON schema for component configuration.
        
        Returns:
            JSON schema dictionary describing valid configuration
        """
        ...
    
    def health_check(self) -> bool:
        """
        Check if component is healthy and ready to process data.
        
        Returns:
            True if component is healthy, False otherwise
        """
        ...
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get component performance metrics.
        
        Returns:
            Dictionary containing performance metrics
        """
        ...

# ===== Document Processing Protocol =====

@runtime_checkable
class DocumentProcessor(BaseComponent, Protocol):
    """
    Protocol for document processing components.
    
    Document processors handle the conversion of raw files into
    structured document data according to the DocumentProcessing contract.
    """
    
    @abstractmethod
    def process(self, input_data: DocumentProcessingInput) -> DocumentProcessingOutput:
        """
        Process documents according to the contract.
        
        Args:
            input_data: Input data conforming to DocumentProcessingInput contract
            
        Returns:
            Output data conforming to DocumentProcessingOutput contract
            
        Raises:
            ProcessingError: If document processing fails
        """
        ...
    
    @abstractmethod
    def process_batch(self, input_batch: List[DocumentProcessingInput]) -> List[DocumentProcessingOutput]:
        """
        Process multiple documents in batch.
        
        Args:
            input_batch: List of input data
            
        Returns:
            List of output data
        """
        ...
    
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported file formats.
        
        Returns:
            List of supported file extensions (e.g., ['.pdf', '.docx'])
        """
        ...
    
    @abstractmethod
    def can_process(self, file_path: str) -> bool:
        """
        Check if this processor can handle the given file.
        
        Args:
            file_path: Path to file to check
            
        Returns:
            True if file can be processed, False otherwise
        """
        ...
    
    @abstractmethod
    def estimate_processing_time(self, input_data: DocumentProcessingInput) -> float:
        """
        Estimate processing time for given input.
        
        Args:
            input_data: Input data to estimate for
            
        Returns:
            Estimated processing time in seconds
        """
        ...

# ===== Chunking Protocol =====

@runtime_checkable
class Chunker(BaseComponent, Protocol):
    """
    Protocol for chunking components.
    
    Chunkers split documents into smaller, manageable pieces
    for downstream processing according to the Chunking contract.
    """
    
    @abstractmethod
    def chunk(self, input_data: ChunkingInput) -> ChunkingOutput:
        """
        Chunk documents according to the contract.
        
        Args:
            input_data: Input data conforming to ChunkingInput contract
            
        Returns:
            Output data conforming to ChunkingOutput contract
            
        Raises:
            ChunkingError: If chunking fails
        """
        ...
    
    @abstractmethod
    def estimate_chunks(self, input_data: ChunkingInput) -> int:
        """
        Estimate number of chunks that will be generated.
        
        Args:
            input_data: Input data to estimate for
            
        Returns:
            Estimated number of chunks
        """
        ...
    
    @abstractmethod
    def supports_content_type(self, content_type: str) -> bool:
        """
        Check if chunker supports the given content type.
        
        Args:
            content_type: Content type to check (e.g., 'text', 'code')
            
        Returns:
            True if content type is supported, False otherwise
        """
        ...
    
    @abstractmethod
    def get_optimal_chunk_size(self, content_type: str) -> int:
        """
        Get the optimal chunk size for a given content type.
        
        Args:
            content_type: Content type
            
        Returns:
            Optimal chunk size in characters
        """
        ...

# ===== Embedding Protocol =====

@runtime_checkable
class Embedder(BaseComponent, Protocol):
    """
    Protocol for embedding components.
    
    Embedders generate vector representations of text chunks
    for similarity search and retrieval according to the Embedding contract.
    """
    
    @abstractmethod
    def embed(self, input_data: EmbeddingInput) -> EmbeddingOutput:
        """
        Generate embeddings according to the contract.
        
        Args:
            input_data: Input data conforming to EmbeddingInput contract
            
        Returns:
            Output data conforming to EmbeddingOutput contract
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        ...
    
    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this embedder."""
        ...
    
    @property
    @abstractmethod
    def max_sequence_length(self) -> int:
        """Get the maximum sequence length this embedder can handle."""
        ...
    
    @abstractmethod
    def supports_batch_processing(self) -> bool:
        """Check if embedder supports batch processing."""
        ...
    
    @abstractmethod
    def get_optimal_batch_size(self) -> int:
        """
        Get the optimal batch size for this embedder.
        
        Returns:
            Optimal batch size
        """
        ...
    
    @abstractmethod
    def estimate_embedding_time(self, input_data: EmbeddingInput) -> float:
        """
        Estimate time to generate embeddings.
        
        Args:
            input_data: Input data to estimate for
            
        Returns:
            Estimated processing time in seconds
        """
        ...

# ===== Graph Enhancement Protocol =====

@runtime_checkable
class GraphEnhancer(BaseComponent, Protocol):
    """
    Protocol for graph enhancement components.
    
    Graph enhancers use graph-based methods to improve embeddings
    by incorporating structural information according to the GraphEnhancement contract.
    """
    
    @abstractmethod
    def enhance(self, input_data: GraphEnhancementInput) -> GraphEnhancementOutput:
        """
        Enhance embeddings using graph methods according to the contract.
        
        Args:
            input_data: Input data conforming to GraphEnhancementInput contract
            
        Returns:
            Output data conforming to GraphEnhancementOutput contract
            
        Raises:
            EnhancementError: If enhancement fails
        """
        ...
    
    @abstractmethod
    def train(self, training_data: GraphEnhancementInput) -> None:
        """
        Train the graph enhancement model.
        
        Args:
            training_data: Training data conforming to GraphEnhancementInput contract
            
        Raises:
            TrainingError: If training fails
        """
        ...
    
    @abstractmethod
    def is_trained(self) -> bool:
        """Check if the model has been trained."""
        ...
    
    @abstractmethod
    def save_model(self, path: str) -> None:
        """
        Save trained model to disk.
        
        Args:
            path: Path to save model
            
        Raises:
            IOError: If model cannot be saved
        """
        ...
    
    @abstractmethod
    def load_model(self, path: str) -> None:
        """
        Load trained model from disk.
        
        Args:
            path: Path to load model from
            
        Raises:
            IOError: If model cannot be loaded
        """
        ...
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Dictionary containing model metadata
        """
        ...
    
    @abstractmethod
    def supports_incremental_training(self) -> bool:
        """Check if the model supports incremental training."""
        ...

# ===== Storage Protocol =====

@runtime_checkable
class Storage(BaseComponent, Protocol):
    """
    Protocol for storage components.
    
    Storage components handle persisting and retrieving
    processed data and embeddings according to the Storage contract.
    """
    
    @abstractmethod
    def store(self, input_data: StorageInput) -> StorageOutput:
        """
        Store data according to the contract.
        
        Args:
            input_data: Input data conforming to StorageInput contract
            
        Returns:
            Output data conforming to StorageOutput contract
            
        Raises:
            StorageError: If storage operation fails
        """
        ...
    
    @abstractmethod
    def query(self, query_data: QueryInput) -> QueryOutput:
        """
        Query stored data according to the contract.
        
        Args:
            query_data: Query data conforming to QueryInput contract
            
        Returns:
            Output data conforming to QueryOutput contract
            
        Raises:
            RetrievalError: If query fails
        """
        ...
    
    @abstractmethod
    def delete(self, item_ids: List[str]) -> bool:
        """
        Delete items by their IDs.
        
        Args:
            item_ids: List of item IDs to delete
            
        Returns:
            True if all items were deleted successfully
        """
        ...
    
    @abstractmethod
    def update(self, item_id: str, data: Dict[str, Any]) -> bool:
        """
        Update an existing item.
        
        Args:
            item_id: ID of item to update
            data: Updated data
            
        Returns:
            True if update was successful
        """
        ...
    
    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get storage statistics.
        
        Returns:
            Dictionary containing storage statistics
        """
        ...
    
    @abstractmethod
    def get_capacity_info(self) -> Dict[str, Any]:
        """
        Get storage capacity information.
        
        Returns:
            Dictionary containing capacity information
        """
        ...
    
    @abstractmethod
    def supports_transactions(self) -> bool:
        """Check if storage supports transactions."""
        ...

# ===== Schema Validation Protocol =====

@runtime_checkable
class SchemaValidator(BaseComponent, Protocol):
    """
    Protocol for schema validation components.
    
    Schema validators ensure data conformity and validation
    throughout the pipeline.
    """
    
    @abstractmethod
    def validate(self, data: Any, schema_name: str) -> bool:
        """
        Validate data against a schema.
        
        Args:
            data: Data to validate
            schema_name: Name of schema to validate against
            
        Returns:
            True if data is valid
        """
        ...
    
    @abstractmethod
    def get_validation_errors(self, data: Any, schema_name: str) -> List[str]:
        """
        Get validation errors for data.
        
        Args:
            data: Data to validate
            schema_name: Name of schema to validate against
            
        Returns:
            List of validation error messages
        """
        ...
    
    @abstractmethod
    def get_available_schemas(self) -> List[str]:
        """
        Get list of available schemas.
        
        Returns:
            List of schema names
        """
        ...

# ===== Database Connection Protocol =====

@runtime_checkable
class DatabaseConnector(BaseComponent, Protocol):
    """
    Protocol for database connection components.
    
    Database connectors provide database connectivity and
    basic operations.
    """
    
    @abstractmethod
    def connect(self) -> bool:
        """
        Establish database connection.
        
        Returns:
            True if connection successful
        """
        ...
    
    @abstractmethod
    def disconnect(self) -> bool:
        """
        Close database connection.
        
        Returns:
            True if disconnection successful
        """
        ...
    
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if database is connected."""
        ...
    
    @abstractmethod
    def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> Any:
        """
        Execute a database query.
        
        Args:
            query: Query to execute
            parameters: Query parameters
            
        Returns:
            Query result
        """
        ...
    
    @abstractmethod
    def get_connection_info(self) -> Dict[str, Any]:
        """
        Get connection information.
        
        Returns:
            Dictionary containing connection details
        """
        ...

# ===== Component Factory Protocol =====

@runtime_checkable
class ComponentFactory(Protocol):
    """
    Protocol for component factories.
    
    Component factories create component instances based on
    configuration and component type.
    """
    
    @abstractmethod
    def create_document_processor(self, name: str, config: Dict[str, Any]) -> "DocumentProcessor":
        """Create a document processor instance."""
        ...
    
    @abstractmethod
    def create_chunker(self, name: str, config: Dict[str, Any]) -> "Chunker":
        """Create a chunker instance."""
        ...
    
    @abstractmethod
    def create_embedder(self, name: str, config: Dict[str, Any]) -> "Embedder":
        """Create an embedder instance."""
        ...
    
    @abstractmethod
    def create_graph_enhancer(self, name: str, config: Dict[str, Any]) -> "GraphEnhancer":
        """Create a graph enhancer instance."""
        ...
    
    @abstractmethod
    def create_storage(self, name: str, config: Dict[str, Any]) -> "Storage":
        """Create a storage instance."""
        ...
    
    @abstractmethod
    def create_schema_validator(self, name: str, config: Dict[str, Any]) -> "SchemaValidator":
        """Create a schema validator instance."""
        ...
    
    @abstractmethod
    def create_database_connector(self, name: str, config: Dict[str, Any]) -> "DatabaseConnector":
        """Create a database connector instance."""
        ...
    
    @abstractmethod
    def create_model_engine(self, name: str, config: Dict[str, Any]) -> "ModelEngine":
        """Create a model engine instance."""
        ...
    
    @abstractmethod
    def list_available_components(self, component_type: ComponentType) -> List[str]:
        """List available component implementations for a type."""
        ...
    
    @abstractmethod
    def get_component_info(self, component_type: ComponentType, name: str) -> Dict[str, Any]:
        """Get information about a specific component implementation."""
        ...


# ===== Model Engine Protocol =====

@runtime_checkable
class ModelEngine(Protocol):
    """
    Protocol for model engine components.
    
    Model engines handle model serving, inference, and resource management.
    They provide the foundation for embedding generation, text completion,
    and other ML model operations.
    """
    
    # Base component properties
    @property
    def name(self) -> str:
        """Component name for identification."""
        ...
    
    @property
    def version(self) -> str:
        """Component version string."""
        ...
    
    @property
    def component_type(self) -> ComponentType:
        """Type of component."""
        ...
    
    def configure(self, config: Dict[str, Any]) -> None:
        """Configure component with parameters."""
        ...
    
    def health_check(self) -> bool:
        """Check component health status."""
        ...
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get component performance metrics."""
        ...
    
    @abstractmethod
    def process(self, input_data: ModelEngineInput) -> ModelEngineOutput:
        """
        Process model inference requests.
        
        Args:
            input_data: Input conforming to ModelEngineInput contract
            
        Returns:
            Output conforming to ModelEngineOutput contract
        """
        ...
    
    @abstractmethod
    def start_server(self) -> bool:
        """
        Start the model server.
        
        Returns:
            True if server started successfully, False otherwise
        """
        ...
    
    @abstractmethod
    def stop_server(self) -> bool:
        """
        Stop the model server.
        
        Returns:
            True if server stopped successfully, False otherwise
        """
        ...
    
    @abstractmethod
    def is_server_running(self) -> bool:
        """
        Check if the model server is running.
        
        Returns:
            True if server is running, False otherwise
        """
        ...
    
    @abstractmethod
    def get_supported_models(self) -> List[str]:
        """
        Get list of supported model names.
        
        Returns:
            List of supported model names
        """
        ...
    
    @abstractmethod
    def estimate_processing_time(self, input_data: ModelEngineInput) -> float:
        """
        Estimate processing time for given input.
        
        Args:
            input_data: Input data to estimate for
            
        Returns:
            Estimated processing time in seconds
        """
        ...