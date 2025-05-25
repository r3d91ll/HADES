"""
Unit tests for the pipeline schemas in the HADES-PathRAG system.

Tests pipeline schema functionality including validation of configuration
parameters, job definitions, and data processing specifications.
"""

import unittest
from pathlib import Path
from pydantic import ValidationError

from src.schemas.pipeline.config import (
    PipelineConfigSchema,
    ProcessingStageConfig,
    InputSourceConfig,
    OutputDestinationConfig,
    DatabaseConfig
)
from src.schemas.pipeline.jobs import (
    JobStatus,
    JobSchema,
    JobResultSchema,
    BatchJobSchema
)
from src.schemas.common.base import BaseSchema


class TestPipelineConfigSchema(unittest.TestCase):
    """Test the PipelineConfigSchema functionality."""
    
    def test_config_instantiation(self):
        """Test that PipelineConfigSchema can be instantiated with default values."""
        # Test with minimal config
        config = PipelineConfigSchema(
            name="test-pipeline",
            version="1.0.0"
        )
        
        self.assertEqual(config.name, "test-pipeline")
        self.assertEqual(config.version, "1.0.0")
        self.assertEqual(config.description, "")  # default value
        self.assertEqual(config.stages, [])  # default value
        self.assertEqual(config.max_workers, 4)  # default value
        self.assertTrue(config.enable_logging)  # default value
        self.assertEqual(config.log_level, "INFO")  # default value
        
        # Test with all attributes
        input_config = InputSourceConfig(type="file", path="/path/to/input")
        output_config = OutputDestinationConfig(type="file", path="/path/to/output")
        db_config = DatabaseConfig(type="arango", connection_string="http://localhost:8529")
        
        stage1 = ProcessingStageConfig(
            name="preprocessing",
            module="src.preprocessing",
            function="preprocess",
            input=input_config,
            output=output_config,
            params={"chunk_size": 1000}
        )
        
        stage2 = ProcessingStageConfig(
            name="embedding",
            module="src.embedding",
            function="embed",
            input=OutputDestinationConfig(type="pipeline", stage="preprocessing"),
            output=output_config,
            params={"model": "all-MiniLM-L6-v2"}
        )
        
        config = PipelineConfigSchema(
            name="full-pipeline",
            version="2.0.0",
            description="Full pipeline configuration",
            stages=[stage1, stage2],
            max_workers=8,
            enable_logging=True,
            log_level="DEBUG",
            database=db_config,
            timeout=300,
            retry_attempts=3,
            metadata={"environment": "production"}
        )
        
        self.assertEqual(config.name, "full-pipeline")
        self.assertEqual(config.version, "2.0.0")
        self.assertEqual(config.description, "Full pipeline configuration")
        self.assertEqual(len(config.stages), 2)
        self.assertEqual(config.stages[0].name, "preprocessing")
        self.assertEqual(config.stages[1].name, "embedding")
        self.assertEqual(config.max_workers, 8)
        self.assertTrue(config.enable_logging)
        self.assertEqual(config.log_level, "DEBUG")
        self.assertEqual(config.database, db_config)
        self.assertEqual(config.timeout, 300)
        self.assertEqual(config.retry_attempts, 3)
        self.assertEqual(config.metadata, {"environment": "production"})
    
    def test_validate_stage_dependencies(self):
        """Test validation of stage dependencies in the pipeline."""
        # Valid pipeline with proper stage dependencies
        input_config = InputSourceConfig(type="file", path="/path/to/input")
        output_config = OutputDestinationConfig(type="file", path="/path/to/output")
        
        stage1 = ProcessingStageConfig(
            name="stage1",
            module="src.module1",
            function="func1",
            input=input_config,
            output=output_config
        )
        
        stage2 = ProcessingStageConfig(
            name="stage2",
            module="src.module2",
            function="func2",
            input=OutputDestinationConfig(type="pipeline", stage="stage1"),
            output=output_config
        )
        
        # This should validate successfully
        config = PipelineConfigSchema(
            name="valid-pipeline",
            version="1.0.0",
            stages=[stage1, stage2]
        )
        
        # Invalid pipeline with reference to non-existent stage
        stage3 = ProcessingStageConfig(
            name="stage3",
            module="src.module3",
            function="func3",
            input=OutputDestinationConfig(type="pipeline", stage="non_existent_stage"),
            output=output_config
        )
        
        # This should raise a validation error
        with self.assertRaises(ValidationError):
            PipelineConfigSchema(
                name="invalid-pipeline",
                version="1.0.0",
                stages=[stage1, stage3]
            )


class TestProcessingStageConfig(unittest.TestCase):
    """Test the ProcessingStageConfig functionality."""
    
    def test_stage_config_instantiation(self):
        """Test that ProcessingStageConfig can be instantiated with required attributes."""
        # Test minimal config
        input_config = InputSourceConfig(type="file", path="/path/to/input")
        output_config = OutputDestinationConfig(type="file", path="/path/to/output")
        
        stage = ProcessingStageConfig(
            name="test-stage",
            module="src.test_module",
            function="test_function",
            input=input_config,
            output=output_config
        )
        
        self.assertEqual(stage.name, "test-stage")
        self.assertEqual(stage.module, "src.test_module")
        self.assertEqual(stage.function, "test_function")
        self.assertEqual(stage.input, input_config)
        self.assertEqual(stage.output, output_config)
        self.assertEqual(stage.params, {})  # default value
        self.assertFalse(stage.enabled)  # default value
        self.assertEqual(stage.timeout, 3600)  # default value
        
        # Test with all attributes
        stage = ProcessingStageConfig(
            name="full-stage",
            module="src.full_module",
            function="full_function",
            input=input_config,
            output=output_config,
            params={"param1": "value1", "param2": 100},
            enabled=True,
            timeout=7200,
            retry_attempts=5,
            description="Full stage configuration",
            dependencies=["other-stage"],
            metadata={"priority": "high"}
        )
        
        self.assertEqual(stage.name, "full-stage")
        self.assertEqual(stage.module, "src.full_module")
        self.assertEqual(stage.function, "full_function")
        self.assertEqual(stage.input, input_config)
        self.assertEqual(stage.output, output_config)
        self.assertEqual(stage.params, {"param1": "value1", "param2": 100})
        self.assertTrue(stage.enabled)
        self.assertEqual(stage.timeout, 7200)
        self.assertEqual(stage.retry_attempts, 5)
        self.assertEqual(stage.description, "Full stage configuration")
        self.assertEqual(stage.dependencies, ["other-stage"])
        self.assertEqual(stage.metadata, {"priority": "high"})
    
    def test_stage_name_validation(self):
        """Test validation of stage name."""
        input_config = InputSourceConfig(type="file", path="/path/to/input")
        output_config = OutputDestinationConfig(type="file", path="/path/to/output")
        
        # Valid stage name
        stage = ProcessingStageConfig(
            name="valid-stage-name",
            module="src.module",
            function="func",
            input=input_config,
            output=output_config
        )
        
        # Invalid stage name (with spaces)
        with self.assertRaises(ValidationError):
            ProcessingStageConfig(
                name="invalid stage name",
                module="src.module",
                function="func",
                input=input_config,
                output=output_config
            )
    
    def test_timeout_validation(self):
        """Test validation of timeout value."""
        input_config = InputSourceConfig(type="file", path="/path/to/input")
        output_config = OutputDestinationConfig(type="file", path="/path/to/output")
        
        # Valid timeout
        stage = ProcessingStageConfig(
            name="stage",
            module="src.module",
            function="func",
            input=input_config,
            output=output_config,
            timeout=60
        )
        
        # Invalid timeout (negative)
        with self.assertRaises(ValidationError):
            ProcessingStageConfig(
                name="stage",
                module="src.module",
                function="func",
                input=input_config,
                output=output_config,
                timeout=-10
            )


class TestInputOutputConfigs(unittest.TestCase):
    """Test the InputSourceConfig and OutputDestinationConfig functionality."""
    
    def test_input_source_config(self):
        """Test that InputSourceConfig can be instantiated with required attributes."""
        # Test file input
        input_config = InputSourceConfig(
            type="file",
            path="/path/to/input"
        )
        
        self.assertEqual(input_config.type, "file")
        self.assertEqual(input_config.path, "/path/to/input")
        self.assertEqual(input_config.format, "auto")  # default value
        
        # Test directory input
        input_config = InputSourceConfig(
            type="directory",
            path="/path/to/input_dir",
            format="json",
            pattern="*.json",
            recursive=True
        )
        
        self.assertEqual(input_config.type, "directory")
        self.assertEqual(input_config.path, "/path/to/input_dir")
        self.assertEqual(input_config.format, "json")
        self.assertEqual(input_config.pattern, "*.json")
        self.assertTrue(input_config.recursive)
        
        # Test database input
        input_config = InputSourceConfig(
            type="database",
            connection="mongodb://localhost:27017",
            collection="documents",
            query={"status": "pending"}
        )
        
        self.assertEqual(input_config.type, "database")
        self.assertEqual(input_config.connection, "mongodb://localhost:27017")
        self.assertEqual(input_config.collection, "documents")
        self.assertEqual(input_config.query, {"status": "pending"})
    
    def test_output_destination_config(self):
        """Test that OutputDestinationConfig can be instantiated with required attributes."""
        # Test file output
        output_config = OutputDestinationConfig(
            type="file",
            path="/path/to/output"
        )
        
        self.assertEqual(output_config.type, "file")
        self.assertEqual(output_config.path, "/path/to/output")
        self.assertEqual(output_config.format, "json")  # default value
        
        # Test directory output
        output_config = OutputDestinationConfig(
            type="directory",
            path="/path/to/output_dir",
            format="csv",
            overwrite=True
        )
        
        self.assertEqual(output_config.type, "directory")
        self.assertEqual(output_config.path, "/path/to/output_dir")
        self.assertEqual(output_config.format, "csv")
        self.assertTrue(output_config.overwrite)
        
        # Test database output
        output_config = OutputDestinationConfig(
            type="database",
            connection="mongodb://localhost:27017",
            collection="processed_documents",
            update_fields=["status", "processed_at"]
        )
        
        self.assertEqual(output_config.type, "database")
        self.assertEqual(output_config.connection, "mongodb://localhost:27017")
        self.assertEqual(output_config.collection, "processed_documents")
        self.assertEqual(output_config.update_fields, ["status", "processed_at"])
        
        # Test pipeline output
        output_config = OutputDestinationConfig(
            type="pipeline",
            stage="preprocessing"
        )
        
        self.assertEqual(output_config.type, "pipeline")
        self.assertEqual(output_config.stage, "preprocessing")
    
    def test_path_validation(self):
        """Test validation of path values."""
        # Valid paths
        InputSourceConfig(type="file", path="input.txt")
        InputSourceConfig(type="file", path="/absolute/path/input.txt")
        InputSourceConfig(type="directory", path="./relative/path")
        
        # Path should be string or Path object
        path_obj = Path("/path/to/file")
        InputSourceConfig(type="file", path=path_obj)
        
        # When type is pipeline, path is not required
        OutputDestinationConfig(type="pipeline", stage="preprocessing")


class TestDatabaseConfig(unittest.TestCase):
    """Test the DatabaseConfig functionality."""
    
    def test_database_config_instantiation(self):
        """Test that DatabaseConfig can be instantiated with required attributes."""
        # Test minimal config
        db_config = DatabaseConfig(
            type="sqlite",
            connection_string="sqlite:///database.db"
        )
        
        self.assertEqual(db_config.type, "sqlite")
        self.assertEqual(db_config.connection_string, "sqlite:///database.db")
        self.assertEqual(db_config.options, {})  # default value
        
        # Test with all attributes
        db_config = DatabaseConfig(
            type="postgres",
            connection_string="postgresql://user:password@localhost:5432/mydb",
            username="user",
            password="password",
            host="localhost",
            port=5432,
            database="mydb",
            options={"pool_size": 10, "ssl": True},
            connection_timeout=30,
            connection_retries=3
        )
        
        self.assertEqual(db_config.type, "postgres")
        self.assertEqual(db_config.connection_string, "postgresql://user:password@localhost:5432/mydb")
        self.assertEqual(db_config.username, "user")
        self.assertEqual(db_config.password, "password")
        self.assertEqual(db_config.host, "localhost")
        self.assertEqual(db_config.port, 5432)
        self.assertEqual(db_config.database, "mydb")
        self.assertEqual(db_config.options, {"pool_size": 10, "ssl": True})
        self.assertEqual(db_config.connection_timeout, 30)
        self.assertEqual(db_config.connection_retries, 3)
    
    def test_arango_specific_config(self):
        """Test Arango-specific configuration."""
        db_config = DatabaseConfig(
            type="arango",
            connection_string="http://localhost:8529",
            database="hades_pathrag",
            collection="documents",
            username="root",
            password="password",
            graph_name="knowledge_graph"
        )
        
        self.assertEqual(db_config.type, "arango")
        self.assertEqual(db_config.connection_string, "http://localhost:8529")
        self.assertEqual(db_config.database, "hades_pathrag")
        self.assertEqual(db_config.collection, "documents")
        self.assertEqual(db_config.username, "root")
        self.assertEqual(db_config.password, "password")
        self.assertEqual(db_config.graph_name, "knowledge_graph")


class TestJobSchemas(unittest.TestCase):
    """Test the job-related schema functionality."""
    
    def test_job_schema_instantiation(self):
        """Test that JobSchema can be instantiated with required attributes."""
        # Test minimal job
        job = JobSchema(
            name="test-job",
            pipeline="test-pipeline"
        )
        
        self.assertEqual(job.name, "test-job")
        self.assertEqual(job.pipeline, "test-pipeline")
        self.assertEqual(job.status, JobStatus.PENDING)  # default value
        self.assertEqual(job.params, {})  # default value
        self.assertIsNotNone(job.id)  # auto-generated
        self.assertIsNotNone(job.created_at)  # auto-generated
        self.assertIsNone(job.started_at)  # default value
        self.assertIsNone(job.completed_at)  # default value
        
        # Test with all attributes
        job = JobSchema(
            id="job-123",
            name="full-job",
            pipeline="full-pipeline",
            status=JobStatus.RUNNING,
            params={"param1": "value1", "param2": 100},
            priority=1,
            created_at="2023-01-01T12:00:00",
            started_at="2023-01-01T12:01:00",
            scheduled_at="2023-01-01T12:00:00",
            max_retries=3,
            retry_count=1,
            timeout=3600,
            owner="admin",
            tags=["important", "batch"],
            metadata={"source": "api"}
        )
        
        self.assertEqual(job.id, "job-123")
        self.assertEqual(job.name, "full-job")
        self.assertEqual(job.pipeline, "full-pipeline")
        self.assertEqual(job.status, JobStatus.RUNNING)
        self.assertEqual(job.params, {"param1": "value1", "param2": 100})
        self.assertEqual(job.priority, 1)
        self.assertEqual(job.created_at, "2023-01-01T12:00:00")
        self.assertEqual(job.started_at, "2023-01-01T12:01:00")
        self.assertEqual(job.scheduled_at, "2023-01-01T12:00:00")
        self.assertEqual(job.max_retries, 3)
        self.assertEqual(job.retry_count, 1)
        self.assertEqual(job.timeout, 3600)
        self.assertEqual(job.owner, "admin")
        self.assertEqual(job.tags, ["important", "batch"])
        self.assertEqual(job.metadata, {"source": "api"})
    
    def test_job_result_schema(self):
        """Test that JobResultSchema can be instantiated with required attributes."""
        # Test success result
        result = JobResultSchema(
            job_id="job-123",
            status=JobStatus.COMPLETED,
            output={"documents_processed": 100}
        )
        
        self.assertEqual(result.job_id, "job-123")
        self.assertEqual(result.status, JobStatus.COMPLETED)
        self.assertEqual(result.output, {"documents_processed": 100})
        self.assertIsNone(result.error)  # default value
        self.assertEqual(result.metrics, {})  # default value
        
        # Test error result
        result = JobResultSchema(
            job_id="job-456",
            status=JobStatus.FAILED,
            error="Connection timeout",
            error_details={"exception": "TimeoutError", "stage": "preprocessing"},
            metrics={"duration": 120.5, "cpu_usage": 85.2}
        )
        
        self.assertEqual(result.job_id, "job-456")
        self.assertEqual(result.status, JobStatus.FAILED)
        self.assertEqual(result.error, "Connection timeout")
        self.assertEqual(result.error_details, {"exception": "TimeoutError", "stage": "preprocessing"})
        self.assertEqual(result.metrics, {"duration": 120.5, "cpu_usage": 85.2})
    
    def test_batch_job_schema(self):
        """Test that BatchJobSchema can be instantiated with required attributes."""
        # Create individual jobs
        job1 = JobSchema(
            name="job1",
            pipeline="test-pipeline"
        )
        
        job2 = JobSchema(
            name="job2",
            pipeline="test-pipeline"
        )
        
        # Test batch job
        batch_job = BatchJobSchema(
            name="batch-test",
            jobs=[job1, job2],
            strategy="parallel"
        )
        
        self.assertEqual(batch_job.name, "batch-test")
        self.assertEqual(len(batch_job.jobs), 2)
        self.assertEqual(batch_job.jobs[0], job1)
        self.assertEqual(batch_job.jobs[1], job2)
        self.assertEqual(batch_job.strategy, "parallel")
        self.assertEqual(batch_job.max_concurrent, 4)  # default value
        
        # Test with all attributes
        batch_job = BatchJobSchema(
            id="batch-123",
            name="full-batch",
            jobs=[job1, job2],
            strategy="sequential",
            max_concurrent=1,
            priority=2,
            owner="admin",
            tags=["batch", "monthly"],
            metadata={"source": "scheduler"},
            created_at="2023-01-01T12:00:00",
            timeout=7200,
            retries_enabled=True
        )
        
        self.assertEqual(batch_job.id, "batch-123")
        self.assertEqual(batch_job.name, "full-batch")
        self.assertEqual(len(batch_job.jobs), 2)
        self.assertEqual(batch_job.strategy, "sequential")
        self.assertEqual(batch_job.max_concurrent, 1)
        self.assertEqual(batch_job.priority, 2)
        self.assertEqual(batch_job.owner, "admin")
        self.assertEqual(batch_job.tags, ["batch", "monthly"])
        self.assertEqual(batch_job.metadata, {"source": "scheduler"})
        self.assertEqual(batch_job.created_at, "2023-01-01T12:00:00")
        self.assertEqual(batch_job.timeout, 7200)
        self.assertTrue(batch_job.retries_enabled)


if __name__ == "__main__":
    unittest.main()
