#!/usr/bin/env python3
"""
Unit tests for GitHub processor.
Uses mocks to avoid GPU and database dependencies during testing.
"""

import unittest
from unittest.mock import Mock, MagicMock, patch, call
import hashlib
from pathlib import Path
import tempfile
import shutil
import json
from datetime import datetime

# Mock the transformers library before import
import sys
sys.modules['transformers'] = MagicMock()
sys.modules['torch'] = MagicMock()
sys.modules['arango'] = MagicMock()
sys.modules['git'] = MagicMock()

from github_processor import GitHubProcessor, get_db_config


class TestGitHubProcessor(unittest.TestCase):
    """Test GitHub processor functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_db_config = {
            'host': 'http://localhost:8529',
            'database': 'test_db',
            'username': 'test_user',
            'password': 'test_pass'
        }
        
        # Create mock database
        self.mock_db = MagicMock()
        self.mock_collection = MagicMock()
        self.mock_db.collection.return_value = self.mock_collection
        self.mock_db.has_collection.return_value = True
        
        # Patch database initialization
        with patch('github_processor.ArangoClient') as mock_client:
            mock_client.return_value.db.return_value = self.mock_db
            self.processor = GitHubProcessor(self.mock_db_config)
    
    def test_init_database(self):
        """Test database initialization."""
        # Database should be initialized
        self.assertIsNotNone(self.processor.db)
        
        # Collections should be checked
        self.mock_db.has_collection.assert_any_call('base_github_repos')
        self.mock_db.has_collection.assert_any_call('base_github_embeddings')
    
    def test_key_generation(self):
        """Test MD5 key generation for paths with special characters."""
        test_cases = [
            ("repo", "src/main.py", 0),
            ("repo", "path/with spaces/file.js", 1),
            ("repo", "path/with/many/slashes/file.go", 2),
            ("user_repo", "file-with-dashes.py", 0),
            ("user_repo", "file.with.dots.js", 0)
        ]
        
        for repo_key, file_path, chunk_index in test_cases:
            key_components = f"{repo_key}_{file_path}_{chunk_index}"
            key_hash = hashlib.md5(key_components.encode()).hexdigest()
            
            # Key should be valid MD5 hash
            self.assertEqual(len(key_hash), 32)
            self.assertTrue(all(c in '0123456789abcdef' for c in key_hash))
            
            # Key should not contain path separators
            self.assertNotIn('/', key_hash)
            self.assertNotIn('\\', key_hash)
    
    @patch('github_processor.git.Repo.clone_from')
    @patch('github_processor.shutil.disk_usage')
    @patch('github_processor.Path.exists')
    def test_clone_repository_success(self, mock_exists, mock_disk_usage, mock_clone):
        """Test successful repository cloning."""
        # Setup mocks
        mock_exists.return_value = False
        mock_disk_usage.return_value = MagicMock(free=2_000_000_000)  # 2GB free
        
        mock_repo = MagicMock()
        mock_repo.active_branch.name = 'main'
        mock_clone.return_value = mock_repo
        
        # Mock file size calculation
        with patch('github_processor.Path.rglob') as mock_rglob:
            mock_file = MagicMock()
            mock_file.is_file.return_value = True
            mock_file.stat.return_value.st_size = 1000
            mock_rglob.return_value = [mock_file] * 100  # 100 files, 1KB each
            
            # Test clone
            result = self.processor.clone_repository("https://github.com/test/repo")
        
        # Verify results
        self.assertEqual(result['status'], 'success')
        self.assertIn('key', result)
        self.assertEqual(result['key'], 'test_repo')
        self.assertAlmostEqual(result['size_mb'], 0.0977, places=2)
        
        # Verify database insert
        self.mock_collection.insert.assert_called_once()
        inserted_doc = self.mock_collection.insert.call_args[0][0]
        self.assertEqual(inserted_doc['_key'], 'test_repo')
        self.assertEqual(inserted_doc['github_url'], 'https://github.com/test/repo')
    
    @patch('github_processor.Path.exists')
    def test_clone_repository_already_exists(self, mock_exists):
        """Test cloning when repository already exists."""
        mock_exists.return_value = True
        
        result = self.processor.clone_repository("https://github.com/test/repo")
        
        self.assertEqual(result['status'], 'exists')
        self.assertEqual(result['key'], 'test_repo')
        
        # Should not attempt to clone
        self.mock_collection.insert.assert_not_called()
    
    @patch('github_processor.shutil.disk_usage')
    @patch('github_processor.Path.exists')
    def test_clone_repository_insufficient_space(self, mock_exists, mock_disk_usage):
        """Test cloning with insufficient disk space."""
        mock_exists.return_value = False
        mock_disk_usage.return_value = MagicMock(free=500_000_000)  # 500MB free
        
        result = self.processor.clone_repository("https://github.com/test/repo")
        
        self.assertEqual(result['status'], 'error')
        self.assertIn('Insufficient disk space', result['error'])
    
    def test_list_code_files(self):
        """Test listing code files in repository."""
        # Setup mock repository document
        self.mock_collection.get.return_value = {
            'clone_path': '/data/repos/test_repo'
        }
        
        # Create temporary directory structure
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            test_files = [
                'main.py',
                'src/module.py',
                'test/test_module.py',
                'docs/readme.md',
                'lib/helper.js',
                '.git/config',  # Should be excluded
                'data.json'  # Should be excluded
            ]
            
            for file_path in test_files:
                full_path = Path(tmpdir) / file_path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.touch()
            
            # Mock the repository path
            self.mock_collection.get.return_value = {
                'clone_path': tmpdir
            }
            
            # Test listing
            files = self.processor.list_code_files('test_repo', ['.py', '.js'])
            
            # Should find only Python and JavaScript files, excluding .git
            expected_files = {'main.py', 'src/module.py', 'test/test_module.py', 'lib/helper.js'}
            found_files = set(files)
            
            self.assertEqual(len(found_files), 4)
            for expected in expected_files:
                self.assertTrue(
                    any(expected in f for f in found_files),
                    f"Expected file {expected} not found in {found_files}"
                )
    
    def test_detect_primary_language(self):
        """Test primary language detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files with different extensions
            files = {
                '.py': 10,  # Most files
                '.js': 5,
                '.md': 3,
                '.txt': 2
            }
            
            for ext, count in files.items():
                for i in range(count):
                    file_path = Path(tmpdir) / f"file{i}{ext}"
                    file_path.touch()
            
            # Test detection
            language = self.processor._detect_primary_language(tmpdir)
            
            self.assertEqual(language, 'Python')
    
    @patch('github_processor.torch.cuda.is_available')
    def test_embedder_initialization(self, mock_cuda):
        """Test embedder initialization (mocked)."""
        mock_cuda.return_value = False  # CPU mode for testing
        
        # Mock the model loading
        with patch('github_processor.AutoModel.from_pretrained') as mock_model:
            with patch('github_processor.AutoTokenizer.from_pretrained') as mock_tokenizer:
                mock_model.return_value = MagicMock()
                mock_tokenizer.return_value = MagicMock()
                
                # Initialize embedder
                self.processor._init_embedder()
                
                # Verify model was loaded with correct parameters
                mock_model.assert_called_once_with(
                    "jinaai/jina-embeddings-v4",
                    trust_remote_code=True,
                    torch_dtype=unittest.mock.ANY
                )
                
                # Verify tokenizer was loaded
                mock_tokenizer.assert_called_once_with("jinaai/jina-embeddings-v4")
                
                # Embedder should be initialized
                self.assertIsNotNone(self.processor.embedder)
                self.assertIsNotNone(self.processor.tokenizer)
    
    def test_get_db_config(self):
        """Test database configuration from environment."""
        with patch.dict('os.environ', {
            'ARANGO_HOST': 'http://test:8529',
            'ARANGO_DATABASE': 'test_db',
            'ARANGO_USERNAME': 'test_user',
            'ARANGO_PASSWORD': 'test_pass'
        }):
            config = get_db_config()
            
            self.assertEqual(config['host'], 'http://test:8529')
            self.assertEqual(config['database'], 'test_db')
            self.assertEqual(config['username'], 'test_user')
            self.assertEqual(config['password'], 'test_pass')
    
    def test_get_db_config_defaults(self):
        """Test database configuration with defaults."""
        with patch.dict('os.environ', {}, clear=True):
            config = get_db_config()
            
            self.assertEqual(config['host'], 'http://192.168.1.69:8529')
            self.assertEqual(config['database'], 'academy_store')
            self.assertEqual(config['username'], 'root')
            self.assertIsNone(config['password'])


class TestWordVecMVP(unittest.TestCase):
    """Test Word2Vec MVP processing workflow."""
    
    @patch('github_processor.GitHubProcessor')
    def test_process_word2vec_mvp(self, mock_processor_class):
        """Test Word2Vec MVP processing (mocked)."""
        from github_processor import process_word2vec_mvp
        
        # Setup mock processor
        mock_processor = MagicMock()
        mock_processor_class.return_value = mock_processor
        
        # Mock successful cloning
        mock_processor.clone_repository.return_value = {
            'status': 'success',
            'key': 'test_repo'
        }
        
        # Mock file listing
        mock_processor.list_code_files.return_value = [
            'main.py',
            'module.py'
        ]
        
        # Mock successful embedding
        mock_processor.embed_files_batch.return_value = {
            'successful': 2,
            'failed': 0
        }
        
        # Run MVP processing
        process_word2vec_mvp()
        
        # Verify all repos were processed
        self.assertEqual(mock_processor.clone_repository.call_count, 4)
        
        # Verify embeddings were generated
        self.assertGreater(mock_processor.embed_files_batch.call_count, 0)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)