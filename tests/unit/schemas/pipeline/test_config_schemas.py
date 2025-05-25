"""
Unit tests for the pipeline configuration schemas in the HADES-PathRAG system.

Tests pipeline configuration components including input sources, output destinations,
database configurations, and processing stages.
"""

import unittest
from pathlib import Path
from typing import Dict, List, Any

from pydantic import ValidationError

from src.schemas.pipeline.config import (
    InputSourceType,
    OutputDestinationType,
    InputSourceConfig,
    OutputDestinationConfig,
    DatabaseType,
    DatabaseConfig,
    ProcessingStageConfig,
    PipelineConfigSchema
)


class TestInputSourceType(unittest.TestCase):
    """Test the InputSourceType enumeration."""
    
    def test_input_source_types(self):
        """Test all defined input source types."""
        expected_types = [
            "file", "directory", "database", "api", "stream", "pipeline"
        ]
        
        # Check that all expected types are defined
        for type_name in expected_types:
            self.assertTrue(hasattr(InputSourceType, type_name.upper()))
            self.assertEqual(getattr(InputSourceType, type_name.upper()).value, type_name)
        
        # Check total number of types
        self.assertEqual(len(InputSourceType), len(expected_types))


class TestOutputDestinationType(unittest.TestCase):
    """Test the OutputDestinationType enumeration."""
    
    def test_output_destination_types(self):
        """Test all defined output destination types."""
        expected_types = [
            "file", "directory", "database", "api", "stream", "pipeline"
        ]
        
        # Check that all expected types are defined
        for type_name in expected_types:
            self.assertTrue(hasattr(OutputDestinationType, type_name.upper()))
            self.assertEqual(getattr(OutputDestinationType, type_name.upper()).value, type_name)
        
        # Check total number of types
        self.assertEqual(len(OutputDestinationType), len(expected_types))


class TestInputSourceConfig(unittest.TestCase):
    """Test the InputSourceConfig functionality."""
    
    def test_config_instantiation(self):
        """Test that InputSourceConfig can be instantiated with required attributes."""
        # Test file source
        source = InputSourceConfig(
            type="file",
            path="/path/to/input.txt"
        )
        
        self.assertEqual(source.type, "file")
        self.assertEqual(source.path, "/path/to/input.txt")
        self.assertEqual(source.format, "auto")  # default value
        self.assertIsNone(source.pattern)  # default value
        self.assertFalse(source.recursive)  # default value
        self.assertIsNone(source.connection)  # default value
        self.assertIsNone(source.collection)  # default value
        self.assertIsNone(source.query)  # default value
        self.assertEqual(source.options, {})  # default value
        
        # Test directory source with all attributes
        source = InputSourceConfig(
            type="directory",
            path="/path/to/directory",
            format="json",
            pattern="*.json",
            recursive=True,
            options={"encoding": "utf-8"}
        )
        
        self.assertEqual(source.type, "directory")
        self.assertEqual(source.path, "/path/to/directory")
        self.assertEqual(source.format, "json")
        self.assertEqual(source.pattern, "*.json")
        self.assertTrue(source.recursive)
        self.assertIsNone(source.connection)
        self.assertIsNone(source.collection)
        self.assertIsNone(source.query)
        self.assertEqual(source.options, {"encoding": "utf-8"})
        
        # Test database source
        source = InputSourceConfig(
            type="database",
            connection="mongodb://localhost:27017",
            collection="documents",
            query={"status": "active"}
        )
        
        self.assertEqual(source.type, "database")
        self.assertIsNone(source.path)
        self.assertEqual(source.format, "auto")
        self.assertEqual(source.connection, "mongodb://localhost:27017")
        self.assertEqual(source.collection, "documents")
        self.assertEqual(source.query, {"status": "active"})
    
    def test_invalid_type(self):
        """Test validation of input source type."""
        with self.assertRaises(ValueError):
            InputSourceConfig(
                type="invalid_type",
                path="/path/to/input.txt"
            )


class TestOutputDestinationConfig(unittest.TestCase):
    """Test the OutputDestinationConfig functionality."""
    
    def test_config_instantiation(self):
        """Test that OutputDestinationConfig can be instantiated with required attributes."""
        # Test file destination
        destination = OutputDestinationConfig(
            type="file",
            path="/path/to/output.json"
        )
        
        self.assertEqual(destination.type, "file")
        self.assertEqual(destination.path, "/path/to/output.json")
        self.assertEqual(destination.format, "json")  # default value
        self.assertFalse(destination.overwrite)  # default value
        self.assertIsNone(destination.connection)  # default value
        self.assertIsNone(destination.collection)  # default value
        self.assertIsNone(destination.update_fields)  # default value
        self.assertEqual(destination.options, {})  # default value
        self.assertIsNone(destination.stage)  # default value
        
        # Test directory destination with all attributes
        destination = OutputDestinationConfig(
            type="directory",
            path="/path/to/output_dir",
            format="csv",
            overwrite=True,
            options={"delimiter": ","}
        )
        
        self.assertEqual(destination.type, "directory")
        self.assertEqual(destination.path, "/path/to/output_dir")
        self.assertEqual(destination.format, "csv")
        self.assertTrue(destination.overwrite)
        self.assertIsNone(destination.connection)
        self.assertIsNone(destination.collection)
        self.assertIsNone(destination.update_fields)
        self.assertEqual(destination.options, {"delimiter": ","})
        
        # Test database destination
        destination = OutputDestinationConfig(
            type="database",
            connection="mongodb://localhost:27017",
            collection="documents",
            update_fields=["status", "updated_at"],
            stage="processed"
        )
        
        self.assertEqual(destination.type, "database")
        self.assertIsNone(destination.path)
        self.assertEqual(destination.format, "json")
        self.assertFalse(destination.overwrite)
        self.assertEqual(destination.connection, "mongodb://localhost:27017")
        self.assertEqual(destination.collection, "documents")
        self.assertEqual(destination.update_fields, ["status", "updated_at"])
        self.assertEqual(destination.stage, "processed")
    
    def test_invalid_type(self):
        """Test validation of output destination type."""
        with self.assertRaises(ValueError):
            OutputDestinationConfig(
                type="invalid_type",
                path="/path/to/output.json"
            )


class TestDatabaseType(unittest.TestCase):
    """Test the DatabaseType enumeration."""
    
    def test_database_types(self):
        """Test all defined database types."""
        expected_types = [
            "sqlite", "postgres", "mysql", "mongodb", 
            "arango", "redis", "elasticsearch"
        ]
        
        # Check that all expected types are defined
        for type_name in expected_types:
            self.assertTrue(hasattr(DatabaseType, type_name.upper()))
            self.assertEqual(getattr(DatabaseType, type_name.upper()).value, type_name)
        
        # Check total number of types
        self.assertEqual(len(DatabaseType), len(expected_types))


class TestDatabaseConfig(unittest.TestCase):
    """Test the DatabaseConfig functionality."""
    
    def test_config_instantiation(self):
        """Test that DatabaseConfig can be instantiated with required attributes."""
        # Test minimal config
        config = DatabaseConfig(
            type="sqlite",
            connection_string="sqlite:///database.db"
        )
        
        self.assertEqual(config.type, "sqlite")
        self.assertEqual(config.connection_string, "sqlite:///database.db")
        self.assertIsNone(config.username)
        self.assertIsNone(config.password)
        self.assertIsNone(config.host)
        self.assertIsNone(config.port)
        self.assertIsNone(config.database)
        self.assertIsNone(config.collection)
        self.assertIsNone(config.graph_name)
        self.assertEqual(config.options, {})
        self.assertIsNone(config.connection_timeout)
        self.assertIsNone(config.connection_retries)
        
        # Test full config
        config = DatabaseConfig(
            type="arango",
            connection_string="http://localhost:8529",
            username="user",
            password="pass",
            host="localhost",
            port=8529,
            database="hades_pathrag",
            collection="documents",
            graph_name="knowledge_graph",
            options={"timeout": 30},
            connection_timeout=5000,
            connection_retries=3
        )
        
        self.assertEqual(config.type, "arango")
        self.assertEqual(config.connection_string, "http://localhost:8529")
        self.assertEqual(config.username, "user")
        self.assertEqual(config.password, "pass")
        self.assertEqual(config.host, "localhost")
        self.assertEqual(config.port, 8529)
        self.assertEqual(config.database, "hades_pathrag")
        self.assertEqual(config.collection, "documents")
        self.assertEqual(config.graph_name, "knowledge_graph")
        self.assertEqual(config.options, {"timeout": 30})
        self.assertEqual(config.connection_timeout, 5000)
        self.assertEqual(config.connection_retries, 3)
    
    def test_invalid_type(self):
        """Test validation of database type."""
        with self.assertRaises(ValueError):
            DatabaseConfig(
                type="invalid_type",
                connection_string="sqlite:///database.db"
            )


class TestProcessingStageConfig(unittest.TestCase):
    """Test the ProcessingStageConfig functionality."""
    
    def test_config_instantiation(self):
        """Test that ProcessingStageConfig can be instantiated with required attributes."""
        # Create input and output configurations
        input_config = InputSourceConfig(
            type="file",
            path="/path/to/input.txt"
        )
        
        output_config = OutputDestinationConfig(
            type="file",
            path="/path/to/output.json"
        )
        
        # Test minimal stage config
        stage = ProcessingStageConfig(
            name="process_text",
            module="src.processors.text",
            function="process_text",
            input=input_config,
            output=output_config
        )
        
        self.assertEqual(stage.name, "process_text")
        self.assertEqual(stage.module, "src.processors.text")
        self.assertEqual(stage.function, "process_text")
        self.assertEqual(stage.input, input_config)
        self.assertEqual(stage.output, output_config)
        self.assertEqual(stage.params, {})
        self.assertFalse(stage.enabled)
        self.assertEqual(stage.timeout, 3600)
        self.assertIsNone(stage.retry_attempts)
        self.assertEqual(stage.description, "")
        self.assertEqual(stage.dependencies, [])
        self.assertEqual(stage.metadata, {})
        
        # Test full stage config
        stage = ProcessingStageConfig(
            name="embed_documents",
            module="src.processors.embedding",
            function="embed_documents",
            input=input_config,
            output=output_config,
            params={"model": "sentence-transformers", "batch_size": 32},
            enabled=True,
            timeout=1800,
            retry_attempts=2,
            description="Embed documents using transformer models",
            dependencies=["load_documents"],
            metadata={"owner": "embedding_team"}
        )
        
        self.assertEqual(stage.name, "embed_documents")
        self.assertEqual(stage.module, "src.processors.embedding")
        self.assertEqual(stage.function, "embed_documents")
        self.assertEqual(stage.input, input_config)
        self.assertEqual(stage.output, output_config)
        self.assertEqual(stage.params, {"model": "sentence-transformers", "batch_size": 32})
        self.assertTrue(stage.enabled)
        self.assertEqual(stage.timeout, 1800)
        self.assertEqual(stage.retry_attempts, 2)
        self.assertEqual(stage.description, "Embed documents using transformer models")
        self.assertEqual(stage.dependencies, ["load_documents"])
        self.assertEqual(stage.metadata, {"owner": "embedding_team"})
    
    def test_invalid_name(self):
        """Test validation of stage name."""
        input_config = InputSourceConfig(type="file", path="/path/to/input.txt")
        output_config = OutputDestinationConfig(type="file", path="/path/to/output.json")
        
        # Test with spaces in name
        with self.assertRaises(ValueError):
            ProcessingStageConfig(
                name="invalid name",
                module="src.processors.text",
                function="process_text",
                input=input_config,
                output=output_config
            )
        
        # Test with special characters in name
        with self.assertRaises(ValueError):
            ProcessingStageConfig(
                name="invalid!name",
                module="src.processors.text",
                function="process_text",
                input=input_config,
                output=output_config
            )
    
    def test_negative_timeout(self):
        """Test validation of timeout value."""
        input_config = InputSourceConfig(type="file", path="/path/to/input.txt")
        output_config = OutputDestinationConfig(type="file", path="/path/to/output.json")
        
        with self.assertRaises(ValueError):
            ProcessingStageConfig(
                name="test_stage",
                module="src.processors.text",
                function="process_text",
                input=input_config,
                output=output_config,
                timeout=-100
            )


class TestPipelineConfigSchema(unittest.TestCase):
    """Test the PipelineConfigSchema functionality."""
    
    def test_config_instantiation(self):
        """Test that PipelineConfigSchema can be instantiated with required attributes."""
        # Test minimal config
        pipeline = PipelineConfigSchema(
            name="test-pipeline",
            version="1.0.0"
        )
        
        self.assertEqual(pipeline.name, "test-pipeline")
        self.assertEqual(pipeline.version, "1.0.0")
        self.assertEqual(pipeline.description, "")
        self.assertEqual(pipeline.stages, [])
        self.assertEqual(pipeline.max_workers, 4)
        self.assertTrue(pipeline.enable_logging)
        self.assertEqual(pipeline.log_level, "INFO")
        self.assertIsNone(pipeline.database)
        self.assertIsNone(pipeline.timeout)
        self.assertIsNone(pipeline.retry_attempts)
        self.assertEqual(pipeline.metadata, {})
        
        # Test with stages
        input_config = InputSourceConfig(type="file", path="/path/to/input.txt")
        intermediate_output = OutputDestinationConfig(type="file", path="/path/to/intermediate.json")
        final_output = OutputDestinationConfig(type="file", path="/path/to/output.json")
        
        stage1 = ProcessingStageConfig(
            name="stage1",
            module="src.processors.text",
            function="process_text",
            input=input_config,
            output=intermediate_output,
            enabled=True
        )
        
        stage2 = ProcessingStageConfig(
            name="stage2",
            module="src.processors.embedding",
            function="embed_documents",
            input=intermediate_output,  # Using output from stage1 as input
            output=final_output,
            dependencies=["stage1"],
            enabled=True
        )
        
        db_config = DatabaseConfig(
            type="arango",
            connection_string="http://localhost:8529",
            database="hades_pathrag"
        )
        
        pipeline = PipelineConfigSchema(
            name="full-pipeline",
            version="2.0.0",
            description="A complete pipeline with multiple stages",
            stages=[stage1, stage2],
            max_workers=8,
            enable_logging=True,
            log_level="DEBUG",
            database=db_config,
            timeout=7200,
            retry_attempts=2,
            metadata={"owner": "pipeline_team"}
        )
        
        self.assertEqual(pipeline.name, "full-pipeline")
        self.assertEqual(pipeline.version, "2.0.0")
        self.assertEqual(pipeline.description, "A complete pipeline with multiple stages")
        self.assertEqual(len(pipeline.stages), 2)
        self.assertEqual(pipeline.stages[0], stage1)
        self.assertEqual(pipeline.stages[1], stage2)
        self.assertEqual(pipeline.max_workers, 8)
        self.assertTrue(pipeline.enable_logging)
        self.assertEqual(pipeline.log_level, "DEBUG")
        self.assertEqual(pipeline.database, db_config)
        self.assertEqual(pipeline.timeout, 7200)
        self.assertEqual(pipeline.retry_attempts, 2)
        self.assertEqual(pipeline.metadata, {"owner": "pipeline_team"})
    
    def test_invalid_dependencies(self):
        """Test validation of stage dependencies."""
        input_config = InputSourceConfig(type="file", path="/path/to/input.txt")
        output_config = OutputDestinationConfig(type="file", path="/path/to/output.json")
        
        stage1 = ProcessingStageConfig(
            name="stage1",
            module="src.processors.text",
            function="process_text",
            input=input_config,
            output=output_config,
            enabled=True
        )
        
        # Stage with non-existent dependency
        stage2 = ProcessingStageConfig(
            name="stage2",
            module="src.processors.embedding",
            function="embed_documents",
            input=input_config,
            output=output_config,
            dependencies=["non_existent_stage"],
            enabled=True
        )
        
        # This should raise a validation error due to the non-existent dependency
        with self.assertRaises(ValueError):
            PipelineConfigSchema(
                name="test-pipeline",
                version="1.0.0",
                stages=[stage1, stage2]
            )


if __name__ == "__main__":
    unittest.main()
