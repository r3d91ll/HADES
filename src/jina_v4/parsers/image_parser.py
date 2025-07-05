"""
Image Parser for Jina v4

Handles multimodal image processing with context extraction and bridge detection.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import numpy as np
from PIL import Image  # type: ignore[import-untyped]
import io
import base64
import re

logger = logging.getLogger(__name__)


class ImageParser:
    """Parse images for multimodal processing with Jina v4."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize image parser with configuration."""
        self.config = config or {}
        self.max_dimension = self.config.get('max_image_dimension', 1024)
        self.enable_ocr = self.config.get('enable_ocr', True)
        self.enable_vision_analysis = self.config.get('enable_vision_analysis', True)
        
    def parse(self, file_path: Path) -> Dict[str, Any]:
        """
        Parse image file and extract multimodal information.
        
        Returns:
            Dictionary containing image data, extracted text, bridges, and metadata
        """
        try:
            # Load and preprocess image
            image = self._load_and_preprocess_image(file_path)
            
            # Classify image type
            image_type = self._classify_image_type(image, file_path)
            
            # Extract visual features
            visual_features = self._extract_visual_features(image, image_type)
            
            # Extract text content if present
            text_content = ""
            if self.enable_ocr:
                text_content = self._extract_text_from_image(image)
            
            # Detect bridges based on content and context
            bridges = self._detect_image_bridges(
                file_path,
                visual_features,
                text_content,
                image_type
            )
            
            # Generate accessible description
            description = self._generate_image_description(
                image,
                visual_features,
                text_content,
                image_type
            )
            
            # Extract metadata
            metadata = self._extract_image_metadata(file_path, image)
            
            # Prepare image for Jina v4 encoding
            image_data = self._prepare_for_jina(image)
            
            return {
                'content': description,  # Text description for text-based processing
                'image_data': image_data,  # Actual image data for Jina v4
                'file_type': 'image',
                'image_type': image_type,
                'text_content': text_content,
                'visual_features': visual_features,
                'bridges': bridges,
                'metadata': metadata,
                'description': description
            }
            
        except Exception as e:
            logger.error(f"Error parsing image {file_path}: {e}")
            return {
                'content': f"[Image: {file_path.name}]",
                'file_type': 'image',
                'image_type': 'unknown',
                'error': str(e)
            }
    
    def _load_and_preprocess_image(self, file_path: Path) -> Image.Image:
        """Load image and preprocess for analysis."""
        image = Image.open(file_path)
        
        # Convert to RGB if necessary
        if image.mode not in ('RGB', 'L'):
            if image.mode == 'RGBA':
                # Create white background for transparency
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[3])
                image = background
            else:
                image = image.convert('RGB')
        
        # Resize if too large
        if max(image.size) > self.max_dimension:
            ratio = self.max_dimension / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        return image
    
    def _classify_image_type(self, image: Image.Image, file_path: Path) -> str:
        """Classify the type/purpose of the image."""
        filename = file_path.name.lower()
        parent_dir = file_path.parent.name.lower()
        
        # Check filename patterns
        if any(pattern in filename for pattern in ['diagram', 'architecture', 'flow']):
            return 'architecture_diagram'
        elif any(pattern in filename for pattern in ['uml', 'class', 'sequence']):
            return 'uml_diagram'
        elif any(pattern in filename for pattern in ['plot', 'graph', 'chart']):
            return 'data_visualization'
        elif any(pattern in filename for pattern in ['screenshot', 'screen', 'ui']):
            return 'screenshot'
        elif any(pattern in filename for pattern in ['mockup', 'wireframe', 'design']):
            return 'ui_mockup'
        elif any(pattern in filename for pattern in ['logo', 'icon', 'avatar']):
            return 'logo_icon'
        elif 'figure' in filename or 'fig' in filename:
            return 'research_figure'
        
        # Check parent directory
        if 'docs' in parent_dir or 'documentation' in parent_dir:
            return 'documentation_image'
        elif 'assets' in parent_dir or 'images' in parent_dir:
            return 'asset_image'
        elif 'output' in parent_dir or 'results' in parent_dir:
            return 'output_visualization'
        
        # Analyze image characteristics
        if self._is_likely_diagram(image):
            return 'diagram'
        elif self._is_likely_screenshot(image):
            return 'screenshot'
        
        return 'general_image'
    
    def _extract_visual_features(self, image: Image.Image, image_type: str) -> Dict[str, Any]:
        """Extract visual features from the image."""
        features = {
            'dimensions': image.size,
            'mode': image.mode,
            'format': image.format if hasattr(image, 'format') else None,
            'image_type': image_type
        }
        
        # Analyze colors
        if image.mode == 'RGB':
            colors = self._analyze_colors(image)
            features['dominant_colors'] = colors
            features['is_monochrome'] = self._is_monochrome(colors)
        
        # Check for specific patterns based on image type
        if image_type in ['architecture_diagram', 'uml_diagram', 'diagram']:
            features['has_boxes'] = self._detect_boxes(image)
            features['has_arrows'] = self._detect_arrows(image)
            features['has_text'] = self._detect_text_regions(image)
        
        elif image_type == 'data_visualization':
            features['has_axes'] = self._detect_axes(image)
            features['has_legend'] = self._detect_legend(image)
            features['chart_type'] = self._detect_chart_type(image)
        
        elif image_type in ['screenshot', 'ui_mockup']:
            features['has_ui_elements'] = self._detect_ui_elements(image)
            features['layout_type'] = self._detect_layout_type(image)
        
        return features
    
    def _extract_text_from_image(self, image: Image.Image) -> str:
        """Extract text from image using OCR."""
        try:
            import pytesseract  # type: ignore[import-not-found]
            
            # Convert to grayscale for better OCR
            if image.mode != 'L':
                gray_image = image.convert('L')
            else:
                gray_image = image
            
            # Extract text
            text: str = pytesseract.image_to_string(gray_image)
            
            # Also get word bounding boxes for better analysis
            data = pytesseract.image_to_data(gray_image, output_type=pytesseract.Output.DICT)
            
            # Filter and clean extracted text
            words = []
            for i, word in enumerate(data['text']):
                if word.strip() and data['conf'][i] > 30:  # Confidence threshold
                    words.append(word.strip())
            
            # Combine both approaches
            if words:
                structured_text = ' '.join(words)
                return f"{text}\n\n[Structured words: {structured_text}]"
            
            return text
            
        except ImportError:
            logger.warning("pytesseract not available, skipping OCR")
            return ""
        except Exception as e:
            logger.warning(f"OCR failed: {e}")
            return ""
    
    def _detect_image_bridges(
        self,
        file_path: Path,
        visual_features: Dict[str, Any],
        text_content: str,
        image_type: str
    ) -> List[Dict[str, Any]]:
        """Detect potential bridges between image and code/docs."""
        bridges = []
        
        # Extract identifiers from text
        if text_content:
            # Look for class names, function names, etc.
            identifiers = self._extract_identifiers(text_content)
            
            for identifier in identifiers:
                if identifier[0].isupper():  # Likely a class name
                    bridges.append({
                        'type': 'visual_class_reference',
                        'source': f"{file_path.name}#{identifier}",
                        'target': identifier,
                        'target_type': 'class',
                        'confidence': 0.7
                    })
                else:  # Likely a function or variable
                    bridges.append({
                        'type': 'visual_function_reference',
                        'source': f"{file_path.name}#{identifier}",
                        'target': identifier,
                        'target_type': 'function',
                        'confidence': 0.6
                    })
        
        # Image type specific bridges
        if image_type == 'architecture_diagram':
            # Architecture diagrams often map to module structure
            bridges.append({
                'type': 'architecture_visualization',
                'source': file_path.name,
                'target_type': 'module_structure',
                'confidence': 0.8
            })
        
        elif image_type == 'uml_diagram':
            # UML diagrams map directly to code structure
            bridges.append({
                'type': 'uml_implementation',
                'source': file_path.name,
                'target_type': 'class_hierarchy',
                'confidence': 0.9
            })
        
        elif image_type == 'data_visualization':
            # Plots often generated by specific code
            bridges.append({
                'type': 'visualization_generator',
                'source': file_path.name,
                'target_type': 'analysis_code',
                'confidence': 0.7
            })
        
        elif image_type == 'screenshot':
            # Screenshots document UI state
            bridges.append({
                'type': 'ui_documentation',
                'source': file_path.name,
                'target_type': 'ui_component',
                'confidence': 0.75
            })
        
        # Path-based bridges
        parent_path = file_path.parent
        if 'notebook' in str(parent_path) or file_path.name.startswith('output_'):
            # Likely generated by a notebook
            bridges.append({
                'type': 'notebook_output',
                'source': file_path.name,
                'target_type': 'jupyter_cell',
                'confidence': 0.8
            })
        
        return bridges
    
    def _generate_image_description(
        self,
        image: Image.Image,
        visual_features: Dict[str, Any],
        text_content: str,
        image_type: str
    ) -> str:
        """Generate a comprehensive text description of the image."""
        parts = []
        
        # Start with image type
        parts.append(f"[{image_type.replace('_', ' ').title()}]")
        
        # Add dimensions
        parts.append(f"Size: {visual_features['dimensions'][0]}x{visual_features['dimensions'][1]}")
        
        # Add visual characteristics
        if visual_features.get('is_monochrome'):
            parts.append("Monochrome image")
        
        if visual_features.get('dominant_colors'):
            colors = ', '.join(visual_features['dominant_colors'][:3])
            parts.append(f"Colors: {colors}")
        
        # Add structural elements based on type
        if image_type in ['architecture_diagram', 'uml_diagram']:
            elements = []
            if visual_features.get('has_boxes'):
                elements.append("boxes/components")
            if visual_features.get('has_arrows'):
                elements.append("arrows/connections")
            if visual_features.get('has_text'):
                elements.append("text labels")
            if elements:
                parts.append(f"Contains: {', '.join(elements)}")
        
        elif image_type == 'data_visualization':
            if visual_features.get('chart_type'):
                parts.append(f"Chart type: {visual_features['chart_type']}")
            if visual_features.get('has_axes'):
                parts.append("Has axes")
            if visual_features.get('has_legend'):
                parts.append("Has legend")
        
        # Add extracted text summary
        if text_content:
            # Get first few meaningful words
            words = text_content.split()[:20]
            text_summary = ' '.join(words)
            if len(words) == 20:
                text_summary += "..."
            parts.append(f"Text: {text_summary}")
        
        return '\n'.join(parts)
    
    def _extract_image_metadata(self, file_path: Path, image: Image.Image) -> Dict[str, Any]:
        """Extract metadata from image file."""
        metadata = {
            'filename': file_path.name,
            'file_size': file_path.stat().st_size,
            'created': file_path.stat().st_ctime,
            'modified': file_path.stat().st_mtime
        }
        
        # Extract EXIF data if available
        if hasattr(image, '_getexif') and image._getexif():
            exif = image._getexif()
            metadata['exif'] = {
                'software': exif.get(0x0131),  # Software used
                'datetime': exif.get(0x0132),  # DateTime
                'artist': exif.get(0x013B),    # Artist/Creator
            }
        
        # Check for embedded metadata
        if hasattr(image, 'info'):
            metadata['embedded'] = image.info
        
        return metadata
    
    def _prepare_for_jina(self, image: Image.Image) -> Dict[str, Any]:
        """Prepare image data for Jina v4 encoding."""
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Also create base64 encoding for API transmission
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return {
            'array': image_array,
            'base64': image_base64,
            'mode': 'RGB',
            'size': image.size
        }
    
    # Helper methods for visual analysis
    
    def _is_likely_diagram(self, image: Image.Image) -> bool:
        """Check if image is likely a diagram based on characteristics."""
        # Convert to grayscale for analysis
        gray = image.convert('L')
        pixels = np.array(gray)
        
        # Diagrams often have high contrast and distinct regions
        std_dev = np.std(pixels)
        unique_colors = len(np.unique(pixels))
        
        return std_dev > 50 and unique_colors < 50
    
    def _is_likely_screenshot(self, image: Image.Image) -> bool:
        """Check if image is likely a screenshot."""
        # Screenshots often have UI patterns and text regions
        # Check for rectangular regions and consistent backgrounds
        pixels = np.array(image)
        
        # Check edges for UI-like patterns
        edge_variance = np.var(pixels[0, :]) + np.var(pixels[-1, :])
        
        return bool(edge_variance < 1000)  # Low variance suggests UI chrome
    
    def _analyze_colors(self, image: Image.Image) -> List[str]:
        """Analyze dominant colors in the image."""
        # Simplified color analysis
        pixels = image.getcolors(maxcolors=100000)
        if not pixels:
            return []
        
        # Sort by frequency
        pixels.sort(key=lambda x: x[0], reverse=True)
        
        # Get top colors as hex
        dominant_colors = []
        for count, color in pixels[:5]:
            if len(color) >= 3:
                hex_color = '#{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2])
                dominant_colors.append(hex_color)
        
        return dominant_colors
    
    def _is_monochrome(self, colors: List[str]) -> bool:
        """Check if image is essentially monochrome."""
        if not colors:
            return True
        
        # Check if all colors are shades of gray
        for color in colors[:3]:  # Check top 3 colors
            r = int(color[1:3], 16)
            g = int(color[3:5], 16)
            b = int(color[5:7], 16)
            
            # If R, G, B values differ significantly, it's not monochrome
            if max(abs(r-g), abs(g-b), abs(r-b)) > 20:
                return False
        
        return True
    
    def _detect_boxes(self, image: Image.Image) -> bool:
        """Detect if image contains box-like shapes (simplified)."""
        # This is a placeholder - real implementation would use edge detection
        return True
    
    def _detect_arrows(self, image: Image.Image) -> bool:
        """Detect if image contains arrows (simplified)."""
        # This is a placeholder - real implementation would use shape detection
        return True
    
    def _detect_text_regions(self, image: Image.Image) -> bool:
        """Detect if image contains text regions."""
        # This is a placeholder - real implementation would use text detection
        return True
    
    def _detect_axes(self, image: Image.Image) -> bool:
        """Detect if image has plot axes."""
        # Look for perpendicular lines at edges
        return True
    
    def _detect_legend(self, image: Image.Image) -> bool:
        """Detect if image has a legend."""
        return True
    
    def _detect_chart_type(self, image: Image.Image) -> str:
        """Detect type of chart/plot."""
        # Simplified detection
        return 'line_plot'  # Placeholder
    
    def _detect_ui_elements(self, image: Image.Image) -> bool:
        """Detect UI elements in screenshot."""
        return True
    
    def _detect_layout_type(self, image: Image.Image) -> str:
        """Detect UI layout type."""
        return 'standard'  # Placeholder
    
    def _extract_identifiers(self, text: str) -> List[str]:
        """Extract potential code identifiers from text."""
        # Simple regex for camelCase and PascalCase
        pattern = r'\b[A-Z][a-zA-Z0-9]*\b|\b[a-z]+(?:[A-Z][a-z]+)*\b'
        identifiers = re.findall(pattern, text)
        
        # Filter out common words
        common_words = {'the', 'and', 'or', 'is', 'in', 'on', 'at', 'to', 'for'}
        return [id for id in identifiers if id.lower() not in common_words and len(id) > 2]