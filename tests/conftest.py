"""
Configure pytest environment.

This file is automatically loaded by pytest and used to set up the test environment.
"""
import os
import sys
import uuid
import pytest
from pathlib import Path

# Add the project root directory to the Python path
# This allows tests to import from the src directory
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.storage.arango.connection import ArangoConnection
from src.types.common import StorageConfig


@pytest.fixture(scope="session")
def test_db_name():
    """Generate a unique test database name to avoid conflicts."""
    return f"hades_test_{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="session")
def arango_connection(test_db_name):
    """Create a connection to a test database with bootstrapped collections.
    
    This fixture creates a fresh test database for the test session and sets up
    the standard collections and graph structure. The database is created with
    a unique name to avoid conflicts with other test runs.
    
    All tests that use this fixture will share the same database throughout
    the test session.
    """
    config = StorageConfig({"database": test_db_name})
    # Force=True ensures clean state even if previous test run crashed
    connection = ArangoConnection.bootstrap(config=config, force=True)
    
    # Return the connection for use in tests
    yield connection
    
    # Optional: Clean up after all tests are done
    # Uncomment if you want the test database to be deleted after tests
    # client = connection.raw_db.client
    # sys_db = client.db("_system")
    # sys_db.delete_database(test_db_name)
