"""
Data preprocessing for Greek motifs
Prepares images and extracts geometric/symbolic features
"""

import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd


class GreekMotifPreprocessor:
    """
    Preprocessor for Greek traditional motifs.
    Handles image standardization, geometric analysis, and metadata extraction.
    """
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (512, 512),
        preserve_aspect: bool = True
    ):
        """
        Initialize preprocessor.
        
        Args:
            target_size: Target image size (width, height)
            preserve_aspect: Whether to preserve aspect ratio
        """
        self.target_size = target_size
        self.preserve_aspect = preserve_aspect
    
    def process_image(
        self,
        image_path: str,
        output_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Process a single motif image.
        
        Args:
            image_path: Path to input image
            output_path: Optional path to save processed image
            
        Returns:
            Processed image as numpy array
        """
        # Load image
        img = Image.open(image_path).convert('RGB')
        
        # Resize
        if self.preserve_aspect:
            img = self._resize_with_aspect(img)
        else:
            img = img.resize(self.target_size, Image.LANCZOS)
        
        # Convert to numpy
        img_array = np.array(img)
        
        # Save if requested
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            Image.fromarray(img_array).save(output_path)
        
        return img_array
    
    def _resize_with_aspect(self, img: Image.Image) -> Image.Image:
        """
        Resize image while preserving aspect ratio.
        Pads with white background.
        """
        # Calculate new size
        aspect = img.width / img.height
        target_aspect = self.target_size[0] / self.target_size[1]
        
        if aspect > target_aspect:
            new_width = self.target_size[0]
            new_height = int(new_width / aspect)
        else:
            new_height = self.target_size[1]
            new_width = int(new_height * aspect)
        
        # Resize
        img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Create white background
        background = Image.new('RGB', self.target_size, (255, 255, 255))
        
        # Paste resized image in center
        x = (self.target_size[0] - new_width) // 2
        y = (self.target_size[1] - new_height) // 2
        background.paste(img, (x, y))
        
        return background
    
    def extract_geometric_features(self, image: np.ndarray) -> Dict:
        """
        Extract geometric features from motif.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary of geometric features
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        features = {}
        
        # Detect symmetry
        features['vertical_symmetry'] = self._check_symmetry(gray, axis='vertical')
        features['horizontal_symmetry'] = self._check_symmetry(gray, axis='horizontal')
        
        # Edge detection for pattern complexity
        edges = cv2.Canny(gray, 50, 150)
        features['edge_density'] = np.sum(edges > 0) / edges.size
        
        # Color analysis
        features['color_histogram'] = self._extract_color_histogram(image)
        features['dominant_colors'] = self._get_dominant_colors(image, k=5)
        
        return features
    
    def _check_symmetry(
        self,
        image: np.ndarray,
        axis: str = 'vertical'
    ) -> float:
        """
        Check image symmetry along an axis.
        
        Returns:
            Symmetry score (0 to 1, higher = more symmetric)
        """
        if axis == 'vertical':
            half1 = image[:, :image.shape[1]//2]
            half2 = np.fliplr(image[:, image.shape[1]//2:])
        else:  # horizontal
            half1 = image[:image.shape[0]//2, :]
            half2 = np.flipud(image[image.shape[0]//2:, :])
        
        # Ensure same size
        min_shape = (min(half1.shape[0], half2.shape[0]),
                     min(half1.shape[1], half2.shape[1]))
        half1 = half1[:min_shape[0], :min_shape[1]]
        half2 = half2[:min_shape[0], :min_shape[1]]
        
        # Calculate similarity
        diff = np.abs(half1.astype(float) - half2.astype(float))
        symmetry = 1.0 - (np.mean(diff) / 255.0)
        
        return symmetry
    
    def _extract_color_histogram(
        self,
        image: np.ndarray,
        bins: int = 32
    ) -> np.ndarray:
        """Extract color histogram features."""
        hist = []
        for i in range(3):  # RGB channels
            channel_hist = cv2.calcHist(
                [image], [i], None, [bins], [0, 256]
            )
            hist.extend(channel_hist.flatten())
        return np.array(hist)
    
    def _get_dominant_colors(
        self,
        image: np.ndarray,
        k: int = 5
    ) -> List[Tuple[int, int, int]]:
        """
        Extract dominant colors using k-means clustering.
        
        Args:
            image: Input image
            k: Number of dominant colors to extract
            
        Returns:
            List of RGB tuples
        """
        pixels = image.reshape(-1, 3)
        pixels = np.float32(pixels)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(
            pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )
        
        centers = np.uint8(centers)
        dominant_colors = [tuple(center) for center in centers]
        
        return dominant_colors
    
    def process_dataset(
        self,
        input_dir: str,
        output_dir: str,
        extract_features: bool = True
    ) -> pd.DataFrame:
        """
        Process entire dataset.
        
        Args:
            input_dir: Input directory with regional subdirectories
            output_dir: Output directory for processed images
            extract_features: Whether to extract geometric features
            
        Returns:
            DataFrame with image metadata and features
        """
        data = []
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # Process each region
        for region_dir in input_path.iterdir():
            if not region_dir.is_dir():
                continue
            
            region_name = region_dir.name
            print(f"Processing region: {region_name}")
            
            # Process each image in region
            for img_file in region_dir.glob("*.png"):
                try:
                    # Output path
                    out_file = output_path / region_name / img_file.name
                    
                    # Process image
                    img_array = self.process_image(str(img_file), str(out_file))
                    
                    # Extract features if requested
                    features = {}
                    if extract_features:
                        features = self.extract_geometric_features(img_array)
                    
                    # Add to dataset
                    data.append({
                        'image_path': str(out_file),
                        'region': region_name,
                        'filename': img_file.name,
                        **features
                    })
                    
                except Exception as e:
                    print(f"Error processing {img_file}: {e}")
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Save metadata
        metadata_path = output_path / "metadata.csv"
        df.to_csv(metadata_path, index=False)
        print(f"Saved metadata to {metadata_path}")
        
        return df


if __name__ == "__main__":
    # Example usage
    preprocessor = GreekMotifPreprocessor(target_size=(512, 512))
    
    # Process dataset
    df = preprocessor.process_dataset(
        input_dir="data/raw",
        output_dir="data/processed",
        extract_features=True
    )
    
    print(f"Processed {len(df)} images")
    print("\nRegional distribution:")
    print(df['region'].value_counts())

