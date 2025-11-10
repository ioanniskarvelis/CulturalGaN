# Phase 2: Symbolic & Geometric Analysis - Complete Guide

## Overview

Phase 2 focuses on **extracting cultural meaning and symbolism** from Greek motifs using AI-powered analysis. This phase creates rich semantic representations that will be used to condition the GAN for authentic motif generation.

---

## What Phase 2 Does

### 1. Symbolic Analysis
- **Analyzes cultural meanings** of each motif
- **Identifies pattern types** (geometric, floral, zoomorphic, symbolic)
- **Documents symbolism** (protection, fertility, prosperity, etc.)
- **Extracts historical context** (time period, usage, craftsmanship)
- **Preserves authenticity markers** (features that define regional style)

### 2. Semantic Embeddings
- **Visual embeddings**: CLIP-based image features
- **Text embeddings**: Semantic understanding from descriptions
- **Geometric embeddings**: Mathematical pattern features
- **Region embeddings**: Geographical origin encoding
- **Combined embeddings**: Multi-modal fusion for GAN conditioning

---

## Phase 2 Workflow

```
┌─────────────────────┐
│ Preprocessed Images │  (from Phase 1)
│   + metadata.csv    │
└──────────┬──────────┘
           │
           v
┌─────────────────────┐
│ Symbolic Analysis   │  ← LLM analyzes cultural meanings
│ - Pattern types     │
│ - Symbolism         │
│ - Historical context│
└──────────┬──────────┘
           │
           v
┌─────────────────────┐
│ annotations.json    │  ← Rich semantic descriptions
└──────────┬──────────┘
           │
           v
┌─────────────────────┐
│ Embedding Creation  │  ← Multi-modal feature extraction
│ - CLIP visual       │
│ - Text semantic     │
│ - Geometric         │
└──────────┬──────────┘
           │
           v
┌─────────────────────┐
│ embeddings.npz      │  ← Ready for GAN training!
│ + metadata.csv      │
└─────────────────────┘
```

---

## Step-by-Step Instructions

### Step 1: Run Symbolic Analysis

#### Option A: Fallback Mode (No API Key Required)

Best for: Initial testing, budget-conscious processing

```bash
python scripts/run_symbolic_analysis.py
```

This uses geometric features and rule-based analysis. Results are basic but functional.

#### Option B: With Anthropic Claude (Recommended) ⭐

Best for: High-quality analysis, cost-effective, excellent cultural understanding

```bash
# Set your API key
export ANTHROPIC_API_KEY=sk-ant-...  # On Windows: $env:ANTHROPIC_API_KEY="sk-ant-..."

# Test with 10 images first
python scripts/run_symbolic_analysis.py --use-vision --limit 10

# Process all images
python scripts/run_symbolic_analysis.py --use-vision
```

Cost estimate: ~$0.003-0.005 per image with Claude 3.5 Sonnet (475 images ≈ $1.50-2.50)

💡 **See detailed setup**: `ANTHROPIC_SETUP.md`

#### Option C: With OpenAI GPT-4 Vision (Alternative)

Best for: If you prefer OpenAI

```bash
# Set your API key
export OPENAI_API_KEY=sk-...

# Test first
python scripts/run_symbolic_analysis.py --use-vision --model gpt-4o --limit 10

# Process all
python scripts/run_symbolic_analysis.py --use-vision --model gpt-4o
```

Cost estimate: ~$0.01-0.02 per image (475 images ≈ $5-10)

### Step 2: Review Results

Check the generated annotations:

```bash
# View annotations summary
python -c "import json; data = json.load(open('data/annotations/annotations.json')); print(f'Total: {len(data)} motifs annotated')"

# View sample annotation
python -c "import json; data = json.load(open('data/annotations/annotations.json')); sample = list(data.values())[0]; print(json.dumps(sample, indent=2))"

# View report
cat data/annotations/symbolic_analysis_report.json
```

### Step 3: Create Semantic Embeddings

```bash
python scripts/create_embeddings.py
```

This will:
- Load annotations
- Extract CLIP visual features
- Create text embeddings from descriptions
- Combine with geometric features
- Save multi-modal embeddings

**Time estimate**: ~5-10 minutes for 500 images on GPU

### Step 4: Verify Embeddings

```bash
# Check embedding dimensions
python -c "import numpy as np; data = np.load('data/embeddings/embeddings.npz'); print('Shapes:'); [print(f'  {k}: {v.shape}') for k, v in data.items()]"

# View statistics
cat data/embeddings/embedding_stats.json
```

---

## Output Files

After Phase 2, you'll have:

```
data/
├── annotations/
│   ├── annotations.json              # Full symbolic analysis
│   └── symbolic_analysis_report.json # Summary statistics
│
└── embeddings/
    ├── embeddings.npz                # All embeddings (compressed)
    ├── embeddings_metadata.csv       # Metadata and descriptions
    └── embedding_stats.json          # Dimension info
```

---

## Understanding the Annotations Format

Each motif has this structure:

```json
{
  "data/processed/Cyclades/image001.png": {
    "filename": "image001.png",
    "region": "Cyclades",
    "geometric_features": {
      "vertical_symmetry": 0.95,
      "horizontal_symmetry": 0.82,
      "edge_density": 0.124
    },
    "symbolic_analysis": {
      "pattern_type": "geometric",
      "geometric_structure": "Bilateral symmetry with repeating meander pattern...",
      "cultural_symbolism": "Represents eternal life and the unending flow...",
      "historical_context": "Common in Cycladic art from 3000-2000 BCE...",
      "color_significance": "Black and red represent earth and vitality...",
      "preservation_notes": "Must preserve exact angular relationships...",
      "authenticity_score": 0.92,
      "key_features": ["meander", "symmetry", "traditional_colors"]
    }
  }
}
```

---

## Embedding Dimensions

The embeddings have these components:

| Component | Dimension | Source |
|-----------|-----------|--------|
| Visual | 512 | CLIP ViT-B/32 |
| Text | 384 | MiniLM sentence encoder |
| Geometric | 16 | Extracted features |
| Region | 11 | One-hot encoding |
| **Combined** | **912** | Concatenated |

The combined embedding will be used for conditional GAN training.

---

## Tips & Best Practices

### 💡 For API-based Analysis

1. **Start small**: Test with `--limit 10` first
2. **Monitor costs**: Track your API usage
3. **Save often**: Checkpoints every 10 images automatically
4. **Resume anytime**: Uses `skip_existing=True` by default
5. **Compare models**: Try both GPT-4 and Claude to see which gives better cultural insights

### 💡 For Fallback Mode

1. **Still useful**: Fallback mode creates valid embeddings
2. **Add manual annotations**: You can edit `annotations.json` manually
3. **Hybrid approach**: Use API for important regions, fallback for others

### 💡 For Embeddings

1. **GPU recommended**: Much faster with CUDA
2. **Check quality**: View sample descriptions in metadata
3. **Dimensionality**: Combined embedding is 912-dim, suitable for GAN conditioning

---

## Troubleshooting

### ❌ "No API key found"

```bash
# Set environment variable
export OPENAI_API_KEY=sk-...          # Linux/Mac
set OPENAI_API_KEY=sk-...             # Windows CMD
$env:OPENAI_API_KEY="sk-..."          # Windows PowerShell
```

### ❌ "annotations.json not found"

You need to run symbolic analysis first:
```bash
python scripts/run_symbolic_analysis.py
```

### ❌ "CLIP model failed to load"

Install CLIP:
```bash
pip install git+https://github.com/openai/CLIP.git
```

### ❌ "Out of memory"

Use CPU or reduce batch size:
```bash
# Force CPU
python scripts/create_embeddings.py --device cpu
```

---

## Cost Estimation

### Anthropic Claude 3.5 Sonnet (Recommended) ⭐
- Cost per image: ~$0.003-0.005
- 475 images: **~$1.50-2.50**
- 1000 images: ~$3-5
- **Fits in free tier!** ($5 free credits)

### OpenAI GPT-4 Vision
- Cost per image: ~$0.01-0.02
- 475 images: ~$5-10
- 1000 images: ~$10-20

### Fallback Mode
- Cost: $0 (free)
- Quality: Basic but functional

---

## Quality Checks

After Phase 2, verify:

✅ **Annotations exist**: `data/annotations/annotations.json`  
✅ **All images processed**: Count matches your dataset  
✅ **Meaningful descriptions**: Sample a few annotations  
✅ **Embeddings created**: `data/embeddings/embeddings.npz`  
✅ **Correct dimensions**: Check embedding_stats.json  
✅ **Regional distribution**: Balanced across regions  

---

## Next Steps

Once Phase 2 is complete:

1. ✅ **Phase 1**: Data preprocessing (completed)
2. ✅ **Phase 2**: Symbolic analysis & embeddings (completed)
3. ⏭️ **Phase 3**: Train StyleGAN3 model
   ```bash
   python src/models/stylegan3_trainer.py
   ```

---

## Advanced: Custom Analysis

You can customize the symbolic analysis prompt by editing:
```python
src/data_processing/symbolic_analysis.py
```

Look for the `_create_analysis_prompt()` method to adjust:
- Analysis categories
- Detail level
- Cultural focus areas
- Output format

---

## Summary

Phase 2 transforms your preprocessed images into **rich semantic representations** that capture both visual appearance and cultural meaning. These embeddings enable the GAN to:

- Generate **authentic** motifs (not generic patterns)
- Respect **regional variations** (Cyclades vs Epirus style)
- Preserve **symbolic meanings** (protection symbols, fertility motifs)
- Maintain **geometric authenticity** (exact pattern structures)

This is the **key difference** from standard GANs - we're not just learning visual patterns, we're encoding **cultural knowledge** into the generation process.

---

**Questions or issues?** Check the main README or open an issue.

**Ready for Phase 3?** Let's train that GAN! 🚀

