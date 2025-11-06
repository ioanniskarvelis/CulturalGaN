# Greek Motif Generation Using GANs: A CDGFD-Inspired Approach

**Cross-Domain Generalization in Greek Fashion Design Using Generative Adversarial Networks**

This project implements a GAN-based methodology for preserving and generating authentic Greek traditional motifs, inspired by the CDGFD (Cross-Domain Generalization in Ethnic Fashion Design) framework. The focus is on **maintaining cultural authenticity** without modern adaptation.

---

## Table of Contents

1. [Overview](#overview)
2. [Methodology](#methodology)
3. [Dataset](#dataset)
4. [Installation](#installation)
5. [Project Structure](#project-structure)
6. [Usage](#usage)
7. [Research Approach](#research-approach)
8. [References](#references)

---

## Overview

This research project aims to:

- **Preserve traditional Greek motifs** through AI-based generation
- **Maintain cultural authenticity** by avoiding modern adaptations
- Apply **GANs and symbolic geometric analysis** for motif generation
- Enable **cross-domain generalization** from traditional artifacts to new designs
- Support **sim-to-real transfer** for practical applications

### Key Features

✓ Authentic Greek motif preservation  
✓ GAN-based generation without modernization  
✓ Symbolic and geometric pattern analysis  
✓ Regional categorization (Aegean, Cyclades, Dodecanese, Epirus, etc.)  
✓ LLM-assisted semantic understanding of motif symbolism  
✓ Cross-domain generalization capabilities  

---

## Methodology

This project follows the **CDGFD framework** adapted for Greek motifs:

### 1. Symbolic and Geometric Analysis

- **Geometric Feature Extraction**: Analyze symmetry, patterns, shapes
- **Symbolic Understanding**: Use LLMs to understand cultural meanings
- **Color Palette Analysis**: Preserve traditional color schemes
- **Regional Variation Mapping**: Document geographical differences

### 2. GAN Architecture

The generation pipeline uses:

- **StyleGAN3** for high-quality motif generation
- **Conditional GANs** for region-specific generation
- **Progressive Growing** for multi-scale pattern learning
- **Feature Matching** to preserve geometric properties

### 3. Cross-Domain Generalization

- **Sim-to-Real Transfer**: From digitized artifacts to new designs
- **Domain Adaptation**: Handle variations in source materials
- **Zero-Shot Generation**: Generate variations of unseen combinations

### 4. Cultural Authenticity Preservation

- No modern color adaptations
- Geometric structures preserved exactly
- Symbolic meanings maintained
- Regional characteristics respected

---

## Dataset

### Greek Traditional Motifs Collection

The dataset is organized by Greek regions:

```
data/raw/
├── Aegean_Islands/      # Aegean maritime patterns
├── Cyclades/            # Cycladic geometric designs
├── Dodecanese/          # Dodecanese floral and geometric
├── Epirus/              # Epirus mountain region motifs
├── Greece/              # General Greek patterns
├── Lesvos/              # Lesvos island patterns
├── North_Aegean/        # North Aegean designs
├── Rhodes/              # Rhodes specific motifs
├── Thessaly/            # Thessaly patterns
├── Thrace/              # Thracian designs
└── Turkey/              # Greek motifs from Anatolia
```

### Dataset Statistics

- **Total Motifs**: 1000+ unique patterns
- **Regions**: 11 geographical areas
- **Formats**: PNG images (512x512 to 2048x2048)
- **Metadata**: Region, period, motif type, symbolism

### Motif Categories

1. **Geometric**: Meanders, spirals, key patterns
2. **Floral**: Roses, carnations, vine patterns
3. **Zoomorphic**: Birds, fish, mythological creatures
4. **Symbolic**: Crosses, stars, protective symbols
5. **Architectural**: Border patterns, friezes

---

## Installation

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (8GB+ VRAM recommended)
- 32GB+ RAM recommended

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/CulturalGaN.git
cd CulturalGaN
```

2. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Download pre-trained models** (when available):
```bash
python scripts/download_models.py
```

---

## Project Structure

```
CulturalGaN/
├── data/
│   ├── raw/                    # Original motif images by region
│   ├── processed/              # Preprocessed and normalized motifs
│   ├── annotations/            # Metadata and symbolic descriptions
│   └── fashion_reference/      # Reference fashion items (if needed)
│
├── src/
│   ├── data_processing/        # Dataset preparation and augmentation
│   ├── models/                 # GAN architectures
│   ├── generation/             # Generation pipelines
│   ├── evaluation/             # Metrics and evaluation
│   └── utils/                  # Helper functions
│
├── scripts/
│   ├── inspect_dataset.py      # Explore dataset statistics
│   ├── normalize_regions.py    # Standardize regional data
│   ├── show_motifs.py          # Visualize motifs
│   └── check_setup.py          # Verify installation
│
├── outputs/                    # Generated motifs (git-ignored)
├── models/                     # Trained model checkpoints (git-ignored)
├── notebooks/                  # Jupyter notebooks for experiments
├── paper/                      # Research paper materials
│
├── dataset.xlsx                # Main dataset spreadsheet (git-ignored)
├── requirements.txt            # Python dependencies
├── CDGFD.pdf                   # Reference methodology paper
└── README.md                   # This file
```

---

## Usage

### 1. Explore the Dataset

```bash
python scripts/inspect_dataset.py
```

This will show:
- Regional distribution
- Motif type statistics
- Sample visualizations

### 2. Visualize Motifs

```bash
python scripts/show_motifs.py --region Cyclades --count 10
```

### 3. Data Preprocessing

```bash
python scripts/normalize_regions.py
```

This will:
- Standardize image sizes
- Extract geometric features
- Generate symbolic descriptions using LLMs
- Create training-ready dataset

### 4. Train GAN Model

```bash
python src/models/train_gan.py \
    --config configs/stylegan3_greek.yaml \
    --region all \
    --epochs 1000
```

### 5. Generate New Motifs

```bash
python src/generation/generate.py \
    --checkpoint models/stylegan3_greek_best.pt \
    --region Cyclades \
    --num_samples 50 \
    --preserve_authenticity true
```

### 6. Evaluate Authenticity

```bash
python src/evaluation/evaluate_authenticity.py \
    --generated outputs/generated/ \
    --reference data/processed/Cyclades/
```

---

## Research Approach

### Phase 1: Data Collection & Analysis (Completed)

✓ Collected 1000+ Greek motifs from 11 regions  
✓ Organized by geographical origin  
✓ Digitized from traditional artifacts  

### Phase 2: Symbolic & Geometric Analysis (In Progress)

- [ ] Extract geometric features (symmetry, patterns)
- [ ] LLM-based semantic analysis of motif meanings
- [ ] Document cultural symbolism per region
- [ ] Create feature embeddings

### Phase 3: GAN Training (Planned)

- [ ] Implement StyleGAN3 architecture
- [ ] Train conditional GAN on regional categories
- [ ] Implement progressive growing strategy
- [ ] Add authenticity-preserving loss functions

### Phase 4: Cross-Domain Generalization (Planned)

- [ ] Sim-to-real transfer validation
- [ ] Zero-shot generation experiments
- [ ] Domain adaptation techniques
- [ ] Evaluation with cultural experts

### Phase 5: Evaluation & Validation (Planned)

- [ ] Quantitative metrics (FID, IS, precision/recall)
- [ ] Cultural authenticity assessment
- [ ] Expert panel evaluation
- [ ] Comparison with original motifs

---

## Key Differences from Modern Adaptation Approaches

This project **DOES NOT**:
- ❌ Modernize traditional patterns
- ❌ Adapt colors to contemporary trends
- ❌ Simplify motifs for commercial appeal
- ❌ Blend traditional with modern aesthetics

This project **DOES**:
- ✅ Preserve exact geometric structures
- ✅ Maintain traditional color palettes
- ✅ Respect cultural symbolism
- ✅ Document regional authenticity
- ✅ Generate new variations within traditional constraints

---

## Technical Requirements

### Hardware

- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3060 or better)
- **RAM**: 32GB+ recommended
- **Storage**: 100GB+ for datasets and models
- **OS**: Windows 10/11, Linux, or macOS

### Software

- Python 3.9+
- PyTorch 2.1+
- CUDA 11.8+ (for GPU acceleration)
- See `requirements.txt` for complete list

### Cloud Alternatives

If local GPU is not available:
- Google Colab Pro ($10/month)
- RunPod (~$0.50/hr for RTX 4090)
- Lambda Labs (~$1.10/hr for A100)

---

## Data Management

### Excluded from Git

The following directories are **ignored** by Git to keep the repository lean:

- `data/` (all subfolders)
- `outputs/` (generated motifs)
- `models/checkpoints/` (trained weights)
- `dataset.xlsx` (large spreadsheet, >100MB)
- `venv/` (Python virtual environment)
- `__pycache__/` (Python cache files)

Use `.gitkeep` files to preserve empty folder structure.

### Large Dataset Files

The `dataset.xlsx` file is >100MB and should remain local. Options:

1. **Keep local** (recommended): Already ignored in `.gitignore`
2. **Convert to chunked format**:
   ```bash
   python scripts/split_xlsx.py dataset.xlsx --out data/raw/dataset --rows 50000
   ```
3. **Use Git LFS** (if versioning required):
   ```bash
   git lfs track "*.xlsx"
   ```

---

## Ethical Considerations

### Cultural Sensitivity

- All motifs are properly documented with cultural origins
- No cultural appropriation or misrepresentation
- Collaboration with Greek cultural experts
- Acknowledgment of traditional artisans
- Respect for cultural heritage

### Research Ethics

- Dataset sources properly credited
- Permissions obtained for copyrighted materials
- Transparent AI methodology
- Clear labeling of AI-generated vs. traditional
- Consideration of impact on traditional crafts

---

## References

### Primary Methodology

- **CDGFD Paper**: Deng, M., & Chen, L. (2025). "CDGFD: Cross-Domain Generalization in Ethnic Fashion Design Using LLMs and GANs: A Symbolic and Geometric Approach." *IEEE Access*, 13, 7192-7207. doi:10.1109/ACCESS.2024.3524444

### GAN Architectures

- Karras, T., et al. (2021). "Alias-Free Generative Adversarial Networks" (StyleGAN3)
- Goodfellow, I., et al. (2014). "Generative Adversarial Networks"
- Mirza, M., & Osindero, S. (2014). "Conditional Generative Adversarial Nets"

### Greek Cultural Heritage

- Skiadarmou, A. "Greek Folk Art"
- Papantoniou, I. "Traditional Greek Costumes"
- Jones, O. "The Grammar of Ornament" (Greek patterns chapter)

---

## Contributing

This is a research project. If you'd like to contribute:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Submit a pull request

Please ensure:
- Cultural sensitivity in all contributions
- Proper documentation
- Code follows existing style
- Tests pass

---

## License

[Specify your license here]

---

## Contact

For questions about this research project:

- **Researcher**: [Your Name]
- **Institution**: University of Thessaly
- **Email**: [Your Email]

---

## Acknowledgments

- CDGFD methodology by Meizhen Deng and Ling Chen
- Greek cultural heritage experts
- Traditional artisans and craft practitioners
- Dataset contributors

---

**Last Updated**: January 2025
