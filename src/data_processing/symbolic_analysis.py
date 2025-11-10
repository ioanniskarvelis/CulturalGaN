"""
Symbolic and Cultural Analysis of Greek Motifs
Uses LLMs to extract cultural meanings and symbolism
"""

import os
import json
import base64
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from PIL import Image
import io


class SymbolicAnalyzer:
    """
    Analyzes Greek motifs for cultural symbolism and meaning using LLMs.
    Follows CDGFD methodology for symbolic understanding.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",  # Anthropic Claude
        use_vision: bool = True
    ):
        """
        Initialize the symbolic analyzer.
        
        Args:
            api_key: API key for LLM service (OpenAI or Anthropic)
            model: Model to use for analysis
            use_vision: Whether to use vision capabilities
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.use_vision = use_vision
        
        # Initialize client based on model
        if "gpt" in model:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
                self.provider = "openai"
            except ImportError:
                print("Warning: OpenAI package not installed. Run: pip install openai")
                self.client = None
        elif "claude" in model:
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key or os.getenv("ANTHROPIC_API_KEY"))
                self.provider = "anthropic"
            except ImportError:
                print("Warning: Anthropic package not installed. Run: pip install anthropic")
                self.client = None
        else:
            self.client = None
            self.provider = None
    
    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 for API."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def analyze_motif(
        self,
        image_path: str,
        region: str,
        geometric_features: Optional[Dict] = None
    ) -> Dict:
        """
        Analyze a single motif for cultural symbolism.
        
        Args:
            image_path: Path to motif image
            region: Greek region (e.g., "Cyclades", "Epirus")
            geometric_features: Optional geometric features from preprocessing
            
        Returns:
            Dictionary containing symbolic analysis
        """
        if not self.client:
            return self._fallback_analysis(region, geometric_features)
        
        # Create detailed prompt
        prompt = self._create_analysis_prompt(region, geometric_features)
        
        # Analyze with vision model
        if self.use_vision:
            analysis = self._analyze_with_vision(image_path, prompt)
        else:
            analysis = self._analyze_text_only(prompt, geometric_features)
        
        return analysis
    
    def _create_analysis_prompt(
        self,
        region: str,
        geometric_features: Optional[Dict] = None
    ) -> str:
        """Create detailed analysis prompt."""
        
        symmetry_info = ""
        if geometric_features:
            v_sym = geometric_features.get('vertical_symmetry', 0)
            h_sym = geometric_features.get('horizontal_symmetry', 0)
            edge_density = geometric_features.get('edge_density', 0)
            
            symmetry_info = f"""
Geometric Features Detected:
- Vertical symmetry: {v_sym:.2f}
- Horizontal symmetry: {h_sym:.2f}
- Pattern complexity (edge density): {edge_density:.3f}
"""
        
        prompt = f"""Analyze this traditional Greek motif from the {region} region.

{symmetry_info}

Please provide a detailed analysis in the following categories:

1. **Pattern Type**: Identify the type of motif (geometric, floral, zoomorphic, symbolic, architectural)

2. **Geometric Structure**: Describe the geometric properties:
   - Symmetry patterns (radial, bilateral, translational)
   - Repetition and rhythm
   - Key geometric shapes

3. **Cultural Symbolism**: Explain the cultural meaning and symbolism:
   - Traditional significance in Greek culture
   - Symbolic elements (protection, fertility, prosperity, etc.)
   - Regional variations and local meanings

4. **Historical Context**: 
   - Time period or era
   - Common usage (clothing, textiles, architecture, pottery)
   - Traditional craftsmanship techniques

5. **Color Significance**: 
   - Traditional color meanings
   - Regional color preferences
   - Symbolic color associations

6. **Preservation Notes**: 
   - Key features that MUST be preserved for authenticity
   - Elements that define this as specifically {region} style
   - Critical geometric or symbolic elements

Format your response as a structured JSON with these keys:
- pattern_type
- geometric_structure
- cultural_symbolism
- historical_context
- color_significance
- preservation_notes
- authenticity_score (0-1, how authentic/traditional it appears)
- key_features (list of critical features)

Be specific to Greek traditional motifs and avoid modern interpretations."""
        
        return prompt
    
    def _analyze_with_vision(self, image_path: str, prompt: str) -> Dict:
        """Analyze motif using vision-language model."""
        
        try:
            if self.provider == "openai":
                # Encode image
                base64_image = self.encode_image(image_path)
                
                # Call OpenAI API
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=2000,
                    temperature=0.3
                )
                
                result_text = response.choices[0].message.content
                
            elif self.provider == "anthropic":
                # Read and encode image
                with open(image_path, "rb") as img_file:
                    image_data = base64.b64encode(img_file.read()).decode("utf-8")
                
                # Determine media type
                media_type = "image/png"
                if image_path.lower().endswith(".jpg") or image_path.lower().endswith(".jpeg"):
                    media_type = "image/jpeg"
                
                # Call Anthropic API
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=2000,
                    temperature=0.3,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": image_data,
                                    },
                                },
                                {
                                    "type": "text",
                                    "text": prompt
                                }
                            ],
                        }
                    ],
                )
                
                result_text = response.content[0].text
            
            else:
                return self._fallback_analysis(None, None)
            
            # Parse JSON response
            try:
                # Extract JSON from markdown code blocks if present
                if "```json" in result_text:
                    result_text = result_text.split("```json")[1].split("```")[0].strip()
                elif "```" in result_text:
                    result_text = result_text.split("```")[1].split("```")[0].strip()
                
                analysis = json.loads(result_text)
            except json.JSONDecodeError:
                # If JSON parsing fails, return raw text
                analysis = {
                    "raw_analysis": result_text,
                    "pattern_type": "unknown",
                    "error": "Failed to parse JSON response"
                }
            
            return analysis
            
        except Exception as e:
            print(f"Error in vision analysis: {e}")
            return self._fallback_analysis(None, None)
    
    def _analyze_text_only(self, prompt: str, geometric_features: Optional[Dict]) -> Dict:
        """Fallback: analyze based on text description only."""
        # This would use just the prompt without vision
        # Simplified implementation
        return self._fallback_analysis(None, geometric_features)
    
    def _fallback_analysis(
        self,
        region: Optional[str],
        geometric_features: Optional[Dict]
    ) -> Dict:
        """
        Fallback analysis when LLM is not available.
        Uses geometric features and regional knowledge.
        """
        analysis = {
            "pattern_type": "geometric",
            "geometric_structure": "Traditional Greek pattern with symmetrical design",
            "cultural_symbolism": f"Traditional motif from {region or 'Greece'}",
            "historical_context": "Greek traditional art",
            "color_significance": "Traditional Greek color palette",
            "preservation_notes": "Preserve geometric structure and color palette",
            "authenticity_score": 0.8,
            "key_features": ["symmetry", "traditional colors", "geometric patterns"],
            "source": "fallback_analysis"
        }
        
        if geometric_features:
            v_sym = geometric_features.get('vertical_symmetry', 0)
            h_sym = geometric_features.get('horizontal_symmetry', 0)
            
            if v_sym > 0.8 and h_sym > 0.8:
                analysis["key_features"].append("high_bilateral_symmetry")
            elif v_sym > 0.8:
                analysis["key_features"].append("vertical_symmetry")
            elif h_sym > 0.8:
                analysis["key_features"].append("horizontal_symmetry")
        
        return analysis
    
    def process_dataset(
        self,
        metadata_path: str,
        output_path: str,
        limit: Optional[int] = None,
        skip_existing: bool = True
    ) -> pd.DataFrame:
        """
        Process entire dataset and generate symbolic annotations.
        
        Args:
            metadata_path: Path to metadata.csv from preprocessing
            output_path: Path to save annotations
            limit: Optional limit on number of images to process
            skip_existing: Skip images that already have annotations
            
        Returns:
            DataFrame with annotations
        """
        # Load metadata
        df = pd.read_csv(metadata_path)
        
        # Load existing annotations if available
        annotations_file = Path(output_path) / "annotations.json"
        if annotations_file.exists() and skip_existing:
            with open(annotations_file, 'r', encoding='utf-8') as f:
                existing_annotations = json.load(f)
        else:
            existing_annotations = {}
        
        annotations = existing_annotations.copy()
        
        # Process each image
        total = limit if limit else len(df)
        processed = 0
        
        for idx, row in df.iterrows():
            if limit and processed >= limit:
                break
            
            image_path = row['image_path']
            
            # Skip if already processed
            if skip_existing and image_path in annotations:
                continue
            
            print(f"Processing {processed+1}/{total}: {row['filename']} ({row['region']})")
            
            # Extract geometric features
            geometric_features = {
                'vertical_symmetry': row.get('vertical_symmetry', 0),
                'horizontal_symmetry': row.get('horizontal_symmetry', 0),
                'edge_density': row.get('edge_density', 0)
            }
            
            # Analyze motif
            analysis = self.analyze_motif(
                image_path=image_path,
                region=row['region'],
                geometric_features=geometric_features
            )
            
            # Add to annotations
            annotations[image_path] = {
                'filename': row['filename'],
                'region': row['region'],
                'geometric_features': geometric_features,
                'symbolic_analysis': analysis
            }
            
            processed += 1
            
            # Save periodically (every 10 images)
            if processed % 10 == 0:
                os.makedirs(output_path, exist_ok=True)
                with open(annotations_file, 'w', encoding='utf-8') as f:
                    json.dump(annotations, f, indent=2, ensure_ascii=False)
                print(f"  → Saved checkpoint at {processed} images")
        
        # Final save
        os.makedirs(output_path, exist_ok=True)
        with open(annotations_file, 'w', encoding='utf-8') as f:
            json.dump(annotations, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Completed! Processed {processed} images")
        print(f"✓ Annotations saved to: {annotations_file}")
        
        return pd.DataFrame.from_dict(annotations, orient='index')
    
    def generate_summary_report(
        self,
        annotations_path: str,
        output_path: str
    ):
        """
        Generate a summary report of symbolic analysis.
        
        Args:
            annotations_path: Path to annotations.json
            output_path: Path to save report
        """
        # Load annotations
        with open(annotations_path, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        
        # Analyze patterns
        pattern_types = {}
        regions = {}
        authenticity_scores = []
        
        for img_path, data in annotations.items():
            analysis = data.get('symbolic_analysis', {})
            region = data.get('region', 'Unknown')
            
            # Count pattern types
            pattern_type = analysis.get('pattern_type', 'unknown')
            pattern_types[pattern_type] = pattern_types.get(pattern_type, 0) + 1
            
            # Count regions
            regions[region] = regions.get(region, 0) + 1
            
            # Collect authenticity scores
            auth_score = analysis.get('authenticity_score', 0)
            if isinstance(auth_score, (int, float)):
                authenticity_scores.append(auth_score)
        
        # Create report
        report = {
            'total_motifs': len(annotations),
            'pattern_types': pattern_types,
            'regional_distribution': regions,
            'average_authenticity': sum(authenticity_scores) / len(authenticity_scores) if authenticity_scores else 0,
            'authenticity_range': {
                'min': min(authenticity_scores) if authenticity_scores else 0,
                'max': max(authenticity_scores) if authenticity_scores else 0
            }
        }
        
        # Save report
        report_path = Path(output_path) / "symbolic_analysis_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Summary Report")
        print(f"  Total motifs analyzed: {report['total_motifs']}")
        print(f"  Pattern types: {dict(sorted(pattern_types.items(), key=lambda x: x[1], reverse=True))}")
        print(f"  Average authenticity: {report['average_authenticity']:.3f}")
        print(f"  Report saved to: {report_path}")
        
        return report


if __name__ == "__main__":
    """
    Example usage:
    
    1. With Vision API (requires API key):
       export OPENAI_API_KEY=your_key_here
       python src/data_processing/symbolic_analysis.py --use-vision
    
    2. Without Vision API (fallback mode):
       python src/data_processing/symbolic_analysis.py
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Symbolic analysis of Greek motifs")
    parser.add_argument("--use-vision", action="store_true", help="Use vision-language model")
    parser.add_argument("--model", default="gpt-4o", help="Model to use (gpt-4o or claude-3-5-sonnet)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images to process")
    parser.add_argument("--metadata", default="data/processed/metadata.csv", help="Path to metadata")
    parser.add_argument("--output", default="data/annotations", help="Output directory")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    print(f"Initializing Symbolic Analyzer...")
    print(f"  Model: {args.model}")
    print(f"  Vision: {args.use_vision}")
    
    analyzer = SymbolicAnalyzer(
        model=args.model,
        use_vision=args.use_vision
    )
    
    # Process dataset
    print(f"\nProcessing dataset...")
    annotations_df = analyzer.process_dataset(
        metadata_path=args.metadata,
        output_path=args.output,
        limit=args.limit,
        skip_existing=True
    )
    
    # Generate summary report
    print(f"\nGenerating summary report...")
    analyzer.generate_summary_report(
        annotations_path=f"{args.output}/annotations.json",
        output_path=args.output
    )
    
    print("\n✓ Symbolic analysis complete!")

