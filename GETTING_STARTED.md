# Getting Started with Greek Motif GAN Project

## Quick Start Guide

This guide will help you get started with the CDGFD-based Greek motif generation project.

---

## Prerequisites

Before you begin, ensure you have:

- ✅ Python 3.9 or higher
- ✅ CUDA-capable GPU (8GB+ VRAM recommended)
- ✅ Git installed
- ✅ 100GB+ free disk space

---

## Installation Steps

### 1. Clone the Repository

```bash
cd CulturalGaN
```

(You're already here!)

### 2. Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Install StyleGAN3 (Required for Training)

```bash
# In a separate directory
cd ..
git clone https://github.com/NVlabs/stylegan3.git
cd stylegan3
pip install -r requirements.txt

# Return to project
cd ../CulturalGaN
```

### 5. Verify Installation

```bash
python scripts/check_setup.py
```

This will check:
- Python version
- PyTorch installation
- CUDA availability
- Required packages

---

## Understanding Your Dataset

### Dataset Structure

Your Greek motifs are organized by region in `data/raw/`:

```
data/raw/
├── Aegean_Islands/      # 23 images
├── Cyclades/            # [check count]
├── Dodecanese/          # [check count]
├── Epirus/              # [check count]
├── Greece/              # [check count]
├── Lesvos/              # [check count]
├── North_Aegean/        # [check count]
├── Rhodes/              # [check count]
├── Thessaly/            # [check count]
├── Thrace/              # [check count]
└── Turkey/              # [check count]
```

### Inspect Your Dataset

```bash
# View dataset statistics
python scripts/inspect_dataset.py

# Show sample motifs from a region
python scripts/show_motifs.py --region Cyclades --count 10

# Normalize and check regions
python scripts/normalize_regions.py
```

---

## Workflow Overview

### Phase 1: Data Preprocessing (Start Here!)

1. **Preprocess images** (resize, normalize, extract features):
   ```bash
   python src/data_processing/preprocess.py
   ```
   
   This will:
   - Resize all images to 512x512
   - Extract geometric features (symmetry, edges)
   - Extract color palettes
   - Save metadata to `data/processed/metadata.csv`

2. **Review processed data**:
   ```bash
   # Check the metadata file
   python -c "import pandas as pd; df = pd.read_csv('data/processed/metadata.csv'); print(df.head()); print(df.info())"
   ```

### Phase 2: Model Setup

1. **Review configuration**:
   ```bash
   # Open and customize if needed
   notepad configs/stylegan3_greek.yaml  # Windows
   nano configs/stylegan3_greek.yaml     # Linux/Mac
   ```

2. **Prepare training environment**:
   - Ensure GPU is available
   - Configure wandb (optional but recommended):
     ```bash
     pip install wandb
     wandb login
     ```

### Phase 3: Training (Coming Soon)

```bash
# Train StyleGAN3 on Greek motifs
python src/models/train_gan.py \
    --config configs/stylegan3_greek.yaml \
    --region all \
    --epochs 1000
```

**Note**: Full training implementation is in development. See `src/models/stylegan3_trainer.py` for the framework.

### Phase 4: Generation (After Training)

```bash
# Generate new motifs
python src/generation/generate_gan.py \
    --checkpoint models/checkpoints/best_model.pt \
    --num_samples 50 \
    --region Cyclades \
    --output_dir outputs/generated
```

---

## Project Structure Explained

### Key Directories

- **`data/`**: Your motif images and metadata
  - `raw/`: Original images (DO NOT MODIFY)
  - `processed/`: Preprocessed images ready for training
  - `annotations/`: Metadata and descriptions

- **`src/`**: Source code
  - `data_processing/`: Image preprocessing and feature extraction
  - `models/`: GAN architectures and trainers
  - `generation/`: Generation pipelines
  - `evaluation/`: Metrics and evaluation (to be implemented)

- **`scripts/`**: Utility scripts for data exploration
  - `inspect_dataset.py`: View dataset statistics
  - `show_motifs.py`: Visualize motifs
  - `normalize_regions.py`: Standardize data

- **`configs/`**: Configuration files
  - `stylegan3_greek.yaml`: Training and model configuration

- **`outputs/`**: Generated results (git-ignored)
- **`models/`**: Trained model weights (git-ignored)

### Key Files

- **`README.md`**: Complete project documentation
- **`MIGRATION_SUMMARY.md`**: What changed in the repository restructure
- **`requirements.txt`**: Python dependencies
- **`CDGFD.pdf`**: Reference methodology paper
- **`.gitignore`**: Files excluded from Git

---

## Common Tasks

### Explore Your Data

```bash
# Count images per region
python scripts/inspect_dataset.py

# View specific motifs
python scripts/show_motifs.py --region Cyclades --count 5

# Check if dataset.xlsx is accessible
python -c "import pandas as pd; df = pd.read_excel('dataset.xlsx'); print(f'Dataset has {len(df)} rows')"
```

### Preprocess Data

```bash
# Process all regions
python src/data_processing/preprocess.py

# Process specific region (modify script as needed)
# Edit src/data_processing/preprocess.py to specify region
```

### Visualize Results

```bash
# After generation
import matplotlib.pyplot as plt
from PIL import Image

img = Image.open('outputs/generated/greek_motif_0000.png')
plt.imshow(img)
plt.axis('off')
plt.show()
```

---

## Troubleshooting

### Common Issues

**1. GPU not detected**
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```
- Install CUDA 11.8+ if not available
- Update GPU drivers

**2. Out of memory errors**
- Reduce `batch_size` in `configs/stylegan3_greek.yaml`
- Close other GPU applications
- Use smaller image resolution (256x256 instead of 512x512)

**3. dataset.xlsx too large to commit**
- Already excluded in `.gitignore`
- Use locally only
- Or split using `scripts/split_xlsx.py`

**4. StyleGAN3 not found**
- Install from official repo: https://github.com/NVlabs/stylegan3
- Follow their installation instructions

---

## Learning Resources

### Understanding the Methodology

1. **Read `CDGFD.pdf`**: The foundation of our approach
2. **Read `README.md`**: Complete project documentation
3. **Review `configs/stylegan3_greek.yaml`**: See all parameters

### Understanding the Code

1. **Data preprocessing**: `src/data_processing/preprocess.py`
2. **GAN training**: `src/models/stylegan3_trainer.py`
3. **Generation**: `src/generation/generate_gan.py`

### StyleGAN3 Resources

- Official repo: https://github.com/NVlabs/stylegan3
- Paper: "Alias-Free Generative Adversarial Networks"
- Pretrained models: Available in StyleGAN3 repo

---

## What's Next?

### Immediate Next Steps

1. ✅ **Verify installation** (run `check_setup.py`)
2. ✅ **Explore dataset** (run `inspect_dataset.py`)
3. ✅ **Preprocess data** (run `preprocess.py`)
4. ⏳ **Implement StyleGAN3 integration** (development in progress)
5. ⏳ **Train model** (after implementation complete)
6. ⏳ **Generate motifs** (after training)

### Development Roadmap

See `MIGRATION_SUMMARY.md` for the complete roadmap:
- Phase 1: Data Preparation (current)
- Phase 2: Model Implementation
- Phase 3: Training
- Phase 4: Evaluation
- Phase 5: Research Documentation

---

## Need Help?

### Documentation

- **Project overview**: `README.md`
- **Migration details**: `MIGRATION_SUMMARY.md`
- **Methodology**: `CDGFD.pdf`
- **This guide**: `GETTING_STARTED.md`

### Code Comments

All new code files have detailed docstrings:
- `src/data_processing/preprocess.py`
- `src/models/stylegan3_trainer.py`
- `src/generation/generate_gan.py`

### Configuration

Check `configs/stylegan3_greek.yaml` for all training parameters and options.

---

## Important Reminders

### This Project Focuses On

✅ **Authentic preservation** of Greek motifs  
✅ **Cultural respect** and sensitivity  
✅ **Traditional patterns** without modification  
✅ **Academic research** methodology  

### This Project Does NOT

❌ Modernize or adapt patterns  
❌ Change traditional colors  
❌ Simplify for commercial purposes  
❌ Mix regions inappropriately  

---

**Ready to begin?** Start with:

```bash
python scripts/inspect_dataset.py
python src/data_processing/preprocess.py
```

Good luck with your research! 🏛️🇬🇷

