"""
Model Format Detection Utility

Provides functionality to detect and validate model file formats before processing.
This prevents format mismatch errors and provides clear guidance to users.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import pickle
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

# Supported model formats
SUPPORTED_FORMATS = {
    '.pth': 'pytorch',
    '.pt': 'pytorch',
    '.pkl': 'pickle', 
    '.pickle': 'pickle',
    '.graphml': 'networkx',
    '.gexf': 'networkx',
    '.json': 'json',
    '.h5': 'hdf5',
    '.hdf5': 'hdf5'
}

# Format compatibility matrix (source -> compatible targets)
FORMAT_COMPATIBILITY = {
    'pytorch': ['pytorch'],  # PyTorch models only work with PyTorch
    'pickle': ['pickle', 'pytorch'],  # Pickle might contain PyTorch models
    'networkx': ['networkx'],  # Graph formats stay as graphs
    'json': ['json'],  # JSON models (metadata, configs)
    'hdf5': ['hdf5', 'pytorch']  # HDF5 can contain various formats
}

# Expected content patterns for validation
CONTENT_VALIDATORS = {
    'pytorch': '_check_pytorch_content',
    'pickle': '_check_pickle_content', 
    'networkx': '_check_networkx_content',
    'json': '_check_json_content'
}

class ModelFormatDetector:
    """Detects and validates model file formats."""
    
    def __init__(self):
        self.detection_cache = {}
        
    def detect_format(self, file_path: str) -> Dict[str, any]:
        """
        Detect the format of a model file.
        
        Args:
            file_path: Path to the model file
            
        Returns:
            dict: Detection results with format, confidence, and metadata
        """
        file_path = Path(file_path)
        
        # Check cache first
        cache_key = str(file_path.absolute())
        if cache_key in self.detection_cache:
            return self.detection_cache[cache_key]
            
        result = {
            'file_path': str(file_path),
            'exists': file_path.exists(),
            'format': None,
            'confidence': 0.0,
            'file_size_mb': 0.0,
            'extension': file_path.suffix.lower(),
            'detected_by': [],
            'validation_errors': [],
            'metadata': {}
        }
        
        if not file_path.exists():
            result['validation_errors'].append(f"File does not exist: {file_path}")
            return result
            
        # Get file size
        try:
            result['file_size_mb'] = file_path.stat().st_size / (1024 * 1024)
        except OSError:
            result['validation_errors'].append("Cannot read file statistics")
            
        # Extension-based detection
        extension = file_path.suffix.lower()
        if extension in SUPPORTED_FORMATS:
            result['format'] = SUPPORTED_FORMATS[extension]
            result['confidence'] = 0.7  # Medium confidence from extension
            result['detected_by'].append('extension')
            
            # Content-based validation
            content_result = self._validate_content(file_path, result['format'])
            result.update(content_result)
        else:
            result['validation_errors'].append(f"Unsupported file extension: {extension}")
            
        # Cache the result
        self.detection_cache[cache_key] = result
        return result
        
    def _validate_content(self, file_path: Path, suspected_format: str) -> Dict[str, any]:
        """Validate file content matches suspected format."""
        result = {
            'content_validated': False,
            'content_errors': []
        }
        
        try:
            if suspected_format in CONTENT_VALIDATORS:
                validator_method = getattr(self, CONTENT_VALIDATORS[suspected_format])
                content_result = validator_method(file_path)
                result.update(content_result)
                
        except Exception as e:
            result['content_errors'].append(f"Content validation failed: {str(e)}")
            
        return result
        
    def _check_pytorch_content(self, file_path: Path) -> Dict[str, any]:
        """Validate PyTorch model file content."""
        result = {'content_validated': False, 'metadata': {}}
        
        try:
            import torch
            
            # Try to load the model metadata without full loading
            with open(file_path, 'rb') as f:
                # Read first few bytes to check for PyTorch magic
                header = f.read(100)
                
            # Check for PyTorch ZIP magic (PyTorch uses ZIP format)
            if header.startswith(b'PK'):
                result['content_validated'] = True
                result['metadata']['pytorch_format'] = 'zip_based'
                result['detected_by'] = result.get('detected_by', []) + ['content']
                
                # Try to get more detailed info without full load
                try:
                    # Load only metadata to avoid memory issues
                    checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
                    
                    if isinstance(checkpoint, dict):
                        result['metadata']['keys'] = list(checkpoint.keys())
                        if 'model_state_dict' in checkpoint:
                            result['metadata']['has_state_dict'] = True
                        if 'optimizer_state_dict' in checkpoint:
                            result['metadata']['has_optimizer'] = True
                            
                except Exception as e:
                    result['content_errors'] = [f"PyTorch load failed: {str(e)}"]
                    
        except ImportError:
            result['content_errors'] = ['PyTorch not available for validation']
            
        return result
        
    def _check_pickle_content(self, file_path: Path) -> Dict[str, any]:
        """Validate pickle file content."""
        result = {'content_validated': False, 'metadata': {}}
        
        try:
            with open(file_path, 'rb') as f:
                # Check pickle magic bytes
                header = f.read(10)
                f.seek(0)
                
                # Pickle files start with specific opcodes
                if header[0:1] in [b'\x80', b'\x00', b'(']:  # Common pickle opcodes
                    result['content_validated'] = True
                    result['detected_by'] = result.get('detected_by', []) + ['content']
                    
                    # Try to peek at the content type
                    try:
                        obj = pickle.load(f)
                        result['metadata']['object_type'] = type(obj).__name__
                        
                        # Check if it's a PyTorch model in pickle format
                        if hasattr(obj, 'state_dict'):
                            result['metadata']['contains_pytorch_model'] = True
                            
                    except Exception as e:
                        result['content_errors'] = [f"Pickle load failed: {str(e)}"]
                        
        except Exception as e:
            result['content_errors'] = [f"File read failed: {str(e)}"]
            
        return result
        
    def _check_networkx_content(self, file_path: Path) -> Dict[str, any]:
        """Validate NetworkX graph file content."""
        result = {'content_validated': False, 'metadata': {}}
        
        try:
            if file_path.suffix.lower() == '.graphml':
                # Parse GraphML XML
                tree = ET.parse(file_path)
                root = tree.getroot()
                
                # Check for GraphML namespace
                if 'graphml' in root.tag.lower():
                    result['content_validated'] = True
                    result['detected_by'] = result.get('detected_by', []) + ['content']
                    
                    # Extract graph metadata
                    graphs = root.findall('.//{http://graphml.graphdrawing.org/xmlns}graph')
                    if graphs:
                        graph = graphs[0]
                        nodes = graph.findall('.//{http://graphml.graphdrawing.org/xmlns}node')
                        edges = graph.findall('.//{http://graphml.graphdrawing.org/xmlns}edge')
                        
                        result['metadata'].update({
                            'node_count': len(nodes),
                            'edge_count': len(edges),
                            'graph_directed': graph.get('edgedefault') == 'directed'
                        })
                        
        except ET.ParseError as e:
            result['content_errors'] = [f"GraphML parse error: {str(e)}"]
        except Exception as e:
            result['content_errors'] = [f"Content validation failed: {str(e)}"]
            
        return result
        
    def _check_json_content(self, file_path: Path) -> Dict[str, any]:
        """Validate JSON model file content."""
        result = {'content_validated': False, 'metadata': {}}
        
        try:
            import json
            
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            result['content_validated'] = True
            result['detected_by'] = result.get('detected_by', []) + ['content']
            result['metadata']['json_keys'] = list(data.keys()) if isinstance(data, dict) else []
            
        except json.JSONDecodeError as e:
            result['content_errors'] = [f"JSON parse error: {str(e)}"]
        except Exception as e:
            result['content_errors'] = [f"Content validation failed: {str(e)}"]
            
        return result
        
    def check_compatibility(self, source_format: str, target_format: str) -> bool:
        """
        Check if source format is compatible with target format.
        
        Args:
            source_format: Source model format
            target_format: Target format requirement
            
        Returns:
            bool: True if compatible
        """
        if source_format not in FORMAT_COMPATIBILITY:
            return False
            
        return target_format in FORMAT_COMPATIBILITY[source_format]
        
    def get_conversion_suggestions(self, source_format: str, target_format: str) -> List[str]:
        """
        Get suggestions for format conversion.
        
        Args:
            source_format: Current model format
            target_format: Desired format
            
        Returns:
            list: Conversion suggestions
        """
        suggestions = []
        
        if source_format == 'networkx' and target_format == 'pytorch':
            suggestions.extend([
                "NetworkX graphs cannot be directly used as PyTorch models",
                "Consider using the graph structure to initialize a PyTorch model",
                "You may need to extract embeddings or features from the graph",
                "Look for corresponding .pth files if this graph represents model structure"
            ])
            
        elif source_format == 'pytorch' and target_format == 'networkx':
            suggestions.extend([
                "PyTorch models need to be converted to graph representation",
                "Extract the model architecture or learned graph structure",
                "Use NetworkX to create graph from model parameters or outputs"
            ])
            
        elif not self.check_compatibility(source_format, target_format):
            suggestions.extend([
                f"Format '{source_format}' is not compatible with '{target_format}'",
                "Check if you have the correct model file",
                "Look for alternative file formats in the same directory",
                "Consult documentation for supported formats"
            ])
            
        return suggestions
        
    def validate_for_use_case(self, file_path: str, use_case: str) -> Dict[str, any]:
        """
        Validate model file for specific use case.
        
        Args:
            file_path: Path to model file
            use_case: Use case ('pytorch_training', 'graph_analysis', etc.)
            
        Returns:
            dict: Validation results with recommendations
        """
        detection_result = self.detect_format(file_path)
        
        result = {
            'valid_for_use_case': False,
            'use_case': use_case,
            'recommendations': [],
            'required_format': None,
            'detected_format': detection_result.get('format'),
            'issues': []
        }
        
        # Define use case requirements
        use_case_requirements = {
            'pytorch_training': 'pytorch',
            'pytorch_inference': 'pytorch', 
            'graph_analysis': 'networkx',
            'isne_training': 'pytorch',
            'isne_bootstrap': 'networkx'
        }
        
        if use_case in use_case_requirements:
            required_format = use_case_requirements[use_case]
            result['required_format'] = required_format
            
            if detection_result.get('format') == required_format:
                result['valid_for_use_case'] = True
                result['recommendations'].append(f"✅ Model format is compatible with {use_case}")
            else:
                result['issues'].append(f"Format mismatch: need '{required_format}', got '{detection_result.get('format')}'")
                result['recommendations'].extend(
                    self.get_conversion_suggestions(detection_result.get('format'), required_format)
                )
                
        else:
            result['issues'].append(f"Unknown use case: {use_case}")
            
        return result


def detect_model_format(file_path: str) -> Dict[str, any]:
    """
    Convenience function to detect model format.
    
    Args:
        file_path: Path to model file
        
    Returns:
        dict: Detection results
    """
    detector = ModelFormatDetector()
    return detector.detect_format(file_path)


def validate_model_for_training(file_path: str) -> bool:
    """
    Quick validation for PyTorch training use case.
    
    Args:
        file_path: Path to model file
        
    Returns:
        bool: True if valid for PyTorch training
    """
    detector = ModelFormatDetector()
    result = detector.validate_for_use_case(file_path, 'pytorch_training')
    return result['valid_for_use_case']


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python model_format_detector.py <model_file>")
        sys.exit(1)
        
    file_path = sys.argv[1]
    detector = ModelFormatDetector()
    result = detector.detect_format(file_path)
    
    print("Model Format Detection Results:")
    print("=" * 40)
    print(f"File: {result['file_path']}")
    print(f"Exists: {result['exists']}")
    print(f"Format: {result['format']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Size: {result['file_size_mb']:.2f} MB")
    print(f"Extension: {result['extension']}")
    
    if result['detected_by']:
        print(f"Detected by: {', '.join(result['detected_by'])}")
        
    if result['validation_errors']:
        print("\\nValidation Errors:")
        for error in result['validation_errors']:
            print(f"  - {error}")
            
    if result['metadata']:
        print("\\nMetadata:")
        for key, value in result['metadata'].items():
            print(f"  {key}: {value}")