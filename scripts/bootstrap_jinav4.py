#!/usr/bin/env python3
"""
Bootstrap script for HADES

Processes documents through Jina v4 and prepares them for ISNE training.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import click
from tqdm import tqdm

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.jina_v4.jina_processor import JinaV4Processor
from src.jina_v4.isne_adapter import convert_jina_to_isne
from src.isne.training.pipeline import ISNETrainingPipeline
from src.storage.arango_client import ArangoClient
from src.utils.filesystem.ignore_handler import IgnoreFileHandler
from src.utils.filesystem.metadata_extractor import FileMetadataExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JinaV4Bootstrap:
    """Bootstrap pipeline for HADES"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.jina = JinaV4Processor(config.get('jina_v4', {}))
        self.storage = ArangoClient(config.get('storage', {}))
        self.ignore_handler = IgnoreFileHandler()
        self.metadata_extractor = FileMetadataExtractor()
        
    async def process_directory(self, directory: Path, extensions: List[str] = None):
        """Process all documents in a directory"""
        
        # Default extensions
        if extensions is None:
            extensions = ['.pdf', '.md', '.py', '.txt', '.ipynb', '.json', '.yaml']
            
        # Find all files
        files = []
        for ext in extensions:
            files.extend(directory.rglob(f'*{ext}'))
            
        # Filter ignored files
        files = [f for f in files if not self.ignore_handler.should_ignore(f)]
        
        logger.info(f"Found {len(files)} files to process")
        
        # Process files with progress bar
        results = []
        for file_path in tqdm(files, desc="Processing documents"):
            try:
                result = await self.process_file(file_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue
                
        return results
    
    async def process_file(self, file_path: Path) -> Dict[str, Any]:
        """Process a single file"""
        
        # Extract filesystem context
        context = self.extract_filesystem_context(file_path)
        
        # Process with Jina v4
        jina_result = await self.jina.process({
            'file_path': str(file_path),
            'hierarchy': context
        })
        
        # Store in database
        await self.store_result(jina_result, context)
        
        return jina_result
    
    def extract_filesystem_context(self, file_path: Path) -> Dict[str, Any]:
        """Extract hierarchical filesystem information"""
        
        parts = file_path.parts
        project_root = self.find_project_root(file_path)
        
        return {
            'full_path': str(file_path),
            'path_array': list(parts),
            'project_root': str(project_root),
            'relative_path': str(file_path.relative_to(project_root)),
            'depth': len(parts),
            'parent_dirs': [str(p) for p in file_path.parents],
            'directory': str(file_path.parent)
        }
    
    def find_project_root(self, path: Path) -> Path:
        """Find project root (git repository or workspace root)"""
        
        current = path if path.is_dir() else path.parent
        
        while current != current.parent:
            if (current / '.git').exists():
                return current
            current = current.parent
            
        return path.parent
    
    async def store_result(self, result: Dict[str, Any], context: Dict[str, Any]):
        """Store processed result in ArangoDB"""
        
        # Store document
        doc_data = {
            '_key': result['document_metadata']['file_path'].replace('/', '_'),
            'content': result.get('content', ''),
            'metadata': result['document_metadata'],
            'hierarchy': context,
            'keywords': result.get('document_keywords', [])
        }
        
        await self.storage.store_document(doc_data)
        
        # Store chunks
        for chunk in result['chunks']:
            chunk_data = {
                **chunk,
                'document_id': doc_data['_key'],
                'hierarchy': context
            }
            await self.storage.store_chunk(chunk_data)
            
    async def train_isne(self, results: List[Dict[str, Any]]):
        """Train ISNE on processed results"""
        
        logger.info("Preparing data for ISNE training")
        
        # Convert to ISNE format
        isne_data = []
        for result in results:
            isne_input = convert_jina_to_isne(result)
            isne_data.append(isne_input)
            
        # Train ISNE
        if self.config.get('train_isne', True):
            logger.info("Training ISNE model")
            pipeline = ISNETrainingPipeline(self.config.get('isne', {}))
            model = await pipeline.train(isne_data)
            
            # Save model
            model_path = Path(self.config.get('output_dir', 'output')) / 'isne_model.pt'
            model.save(model_path)
            logger.info(f"ISNE model saved to {model_path}")


@click.command()
@click.argument('directory', type=click.Path(exists=True, file_okay=False))
@click.option('--config', type=click.Path(exists=True), help='Configuration file')
@click.option('--output-dir', type=click.Path(), default='output', help='Output directory')
@click.option('--no-train', is_flag=True, help='Skip ISNE training')
@click.option('--extensions', multiple=True, help='File extensions to process')
def main(directory, config, output_dir, no_train, extensions):
    """Bootstrap HADES with documents from DIRECTORY"""
    
    # Load configuration
    if config:
        with open(config) as f:
            if config.endswith('.json'):
                config_data = json.load(f)
            else:
                import yaml
                config_data = yaml.safe_load(f)
    else:
        config_data = {}
        
    # Override with command line options
    config_data['output_dir'] = output_dir
    config_data['train_isne'] = not no_train
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Run bootstrap
    bootstrap = JinaV4Bootstrap(config_data)
    results = asyncio.run(bootstrap.process_directory(Path(directory), list(extensions)))
    
    # Save results
    results_path = Path(output_dir) / 'bootstrap_results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'total_documents': len(results),
            'config': config_data,
            'summary': {
                'total_chunks': sum(len(r['chunks']) for r in results),
                'total_keywords': sum(len(r.get('keywords', [])) for r in results)
            }
        }, f, indent=2)
    
    logger.info(f"Bootstrap complete. Results saved to {results_path}")
    
    # Train ISNE if requested
    if not no_train:
        asyncio.run(bootstrap.train_isne(results))


if __name__ == '__main__':
    main()