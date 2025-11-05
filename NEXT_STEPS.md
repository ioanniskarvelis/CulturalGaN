# CulturalGaN - Next Steps & Action Plan

**Last Updated:** November 5, 2025  
**Current Phase:** Phase 1 - Preparation (Weeks 1-3)  
**Project Timeline:** 4-5 months total (16-20 weeks)

---

## Current Project Status

### ✅ Completed Setup
- [x] Project structure created
- [x] Virtual environment set up
- [x] Dependencies installed (PyTorch, Diffusers, etc.)
- [x] Basic generation pipeline implemented (`src/generation/pipeline.py`)
- [x] Directory structure created (`data/`, `outputs/`, `models/`)
- [x] Configuration utilities in place (`src/utils/config.py`)
- [x] Smoke test script ready (`scripts/smoke_test.py`)
- [x] Comprehensive methodology documented in README.md

### ❌ Not Yet Started
- [ ] Greek motif dataset collection and organization
- [ ] Image preprocessing and segmentation
- [ ] Metadata annotation system
- [ ] Model training pipeline
- [ ] Evaluation framework
- [ ] Notebooks for experimentation

---

## IMMEDIATE NEXT STEPS (This Week)

### Step 1: Test Your Setup (30 minutes)

Before diving into data collection, verify that your environment works:

```bash
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Run smoke test (generates a test image using Stable Diffusion)
python scripts\smoke_test.py --garment "dress" --adapt 0.5 --color "modernized"
```

This will:
- Download Stable Diffusion model (~4GB, first time only)
- Generate a test fashion design
- Save output to `outputs/generated_designs/smoke_output.png`

**Expected time:** 5-10 minutes for first run (includes model download)

---

### Step 2: Gather Your Greek Motif Dataset (Priority #1)

According to your methodology (README line 184), you need:
- **Minimum:** 500 unique motifs
- **Optimal:** 1000+ motifs  
- **With augmentation:** 5000+ training images

#### 2.1 Source Your Motifs

**Where to find Greek traditional motifs:**

1. **Museums & Digital Archives**
   - Benaki Museum Digital Collections
   - National Archaeological Museum of Athens
   - Museum of Greek Folk Art
   - Europeana (Greek cultural heritage)
   - Greek Folk Art Museum collections

2. **Academic Resources**
   - University libraries with Greek studies departments
   - Digital humanities projects on Greek culture
   - Archaeological databases

3. **Photography & Documentation**
   - If you have access to physical items (textiles, pottery, wood carvings)
   - Visit local museums with permission to photograph
   - Family heirlooms or local craftspeople

4. **Public Domain & Creative Commons**
   - Wikimedia Commons (search: "Greek embroidery", "Greek pottery patterns")
   - Internet Archive
   - Public domain books on Greek folk art

#### 2.2 Organize Your Collection

Place images in `data/raw/` with this structure:

```
data/raw/
├── crete/
│   ├── floral/
│   ├── geometric/
│   └── zoomorphic/
├── peloponnese/
│   └── ...
├── islands/
│   └── ...
└── ...
```

**Naming convention:**
```
{region}_{type}_{number}_{source}.{ext}
Example: crete_floral_001_benaki.png
```

---

### Step 3: Create Annotation Template (1-2 hours)

Create a JSON template for each motif. See example in `data/annotations/template.json`:

```json
{
  "motif_id": "CRT_001",
  "name": "Cretan Rose Border",
  "region": "Crete",
  "period": "19th century",
  "type": "floral",
  "original_medium": "embroidery",
  "cultural_significance": "Wedding textile decoration",
  "color_palette": {
    "dominant": ["#8B0000", "#2F4F4F"],
    "accent": ["#FFD700", "#FFFFFF"]
  },
  "complexity_score": 7.5,
  "symmetry_type": "bilateral",
  "recommended_garments": ["dress", "blouse", "scarf"],
  "source": "Benaki Museum Collection",
  "rights": "Public Domain",
  "file_path": "data/raw/crete/floral/crete_floral_001.png"
}
```

---

## PHASE 1 DETAILED CHECKLIST (Weeks 1-3)

### Week 1: Dataset Foundation

**Day 1-2: Data Collection Planning**
- [ ] Identify 5-10 reliable sources for Greek motifs
- [ ] Request permissions if needed for copyrighted images
- [ ] Set up download/acquisition workflow
- [ ] Create spreadsheet to track collection progress

**Day 3-5: Initial Collection**
- [ ] Collect first 100 motif images
- [ ] Ensure diversity across:
  - [ ] Regions (Crete, Peloponnese, Islands, Macedonia, etc.)
  - [ ] Types (geometric, floral, zoomorphic, symbolic)
  - [ ] Periods (Byzantine, Ottoman, 19th C, 20th C)
  - [ ] Media (embroidery, carving, pottery, metalwork)

**Day 6-7: Organization & Validation**
- [ ] Organize images into folder structure
- [ ] Verify image quality (min 512x512px)
- [ ] Begin basic metadata documentation
- [ ] Create initial annotation JSON files

---

### Week 2: Preprocessing & Augmentation

**Prerequisites:**
- At least 100-200 motifs collected
- Images organized in `data/raw/`

**Tasks:**

1. **Image Segmentation** (isolate motifs from backgrounds)
   - [ ] Install Segment Anything Model (SAM) if needed
   - [ ] Create segmentation script in `src/data_processing/segment_motifs.py`
   - [ ] Process images to extract clean motifs
   - [ ] Save to `data/processed/`

2. **Quality Enhancement**
   - [ ] Upscale images to consistent resolution (1024x1024px)
   - [ ] Color correction and normalization
   - [ ] Background removal (transparent/white)

3. **Augmentation**
   - [ ] Implement augmentation in `src/data_processing/augmentation.py`
   - [ ] Generate variations:
     - [ ] Rotations (90°, 180°, 270°)
     - [ ] Horizontal/vertical flips
     - [ ] Scale variations (75%, 100%, 125%)
   - [ ] Target: 5x original dataset size

4. **Contemporary Fashion References**
   - [ ] Collect 500-1000 contemporary fashion images
   - [ ] Categories: dresses, tops, accessories
   - [ ] Save to `data/fashion_reference/`

---

### Week 3: Initial Testing & Validation

**Test the Full Pipeline:**

1. **Create First Notebook** (`notebooks/01_data_exploration.ipynb`)
   - [ ] Load and visualize collected motifs
   - [ ] Analyze dataset statistics
   - [ ] Check quality and diversity
   - [ ] Document any issues

2. **Run Initial Generations**
   - [ ] Test pipeline with 10-20 motifs
   - [ ] Try different adaptation levels
   - [ ] Try different garment types
   - [ ] Generate 50-100 test designs

3. **Evaluate & Iterate**
   - [ ] Review generated designs
   - [ ] Identify common problems
   - [ ] Adjust parameters
   - [ ] Document findings

**Deliverables by End of Week 3:**
- [ ] 200-500 unique motifs collected and organized
- [ ] 1000-2500 augmented variations
- [ ] Metadata annotations for all motifs
- [ ] 50-100 test generation outputs
- [ ] Data exploration notebook with visualizations
- [ ] Issues/learnings document

---

## Quick Start Commands

### Activate Environment
```powershell
.\venv\Scripts\Activate.ps1
```

### Run Smoke Test
```bash
python scripts\smoke_test.py
```

### Run with Your Own Motif
```bash
python scripts\smoke_test.py --input "data/raw/your_motif.png" --output "outputs/test_1.png" --garment "dress" --adapt 0.5
```

### Start Jupyter Notebook (if installed)
```bash
jupyter notebook notebooks/
```

---

## Resources & Tools

### For Data Collection
- **Benaki Museum**: https://www.benaki.org/index.php?lang=en
- **Europeana**: https://www.europeana.eu/ (filter by Greece)
- **Wikimedia Commons**: https://commons.wikimedia.org/

### For Image Processing
- **SAM (Segment Anything)**: `pip install segment-anything`
- **Background Removal**: `pip install rembg`
- **Image Enhancement**: Already have OpenCV, Pillow, scikit-image

### For Annotation
- **Roboflow**: https://roboflow.com (free tier)
- **CVAT**: https://cvat.ai (open source)
- **Label Studio**: https://labelstud.io

---

## Common Issues & Solutions

### Issue: Model Download Fails
**Solution:** Set Hugging Face cache directory
```python
import os
os.environ['HF_HOME'] = 'C:/Users/Ioannis/.cache/huggingface'
```

### Issue: CUDA Out of Memory
**Solution:** Reduce batch size or use CPU
```python
# In pipeline.py, it already falls back to CPU if no GPU
```

### Issue: Slow Generation on CPU
**Solution:** Expected 2-5 minutes per image on CPU. Consider:
- Google Colab with free GPU
- RunPod or Vast.ai for cheap GPU access

---

## Timeline Reminder

You're currently in **Phase 1** (Weeks 1-3). Here's what comes next:

- **Phase 2** (Weeks 4-7): Model Training - Fine-tune on your dataset
- **Phase 3** (Weeks 8-11): Generation & Iteration - Create full portfolio
- **Phase 4** (Weeks 12-15): Evaluation - Expert panel, user study
- **Phase 5** (Weeks 16-19): Paper Writing - Publication-ready manuscript

**Total:** ~5 months to complete research project

---

## Questions or Stuck?

If you encounter issues:
1. Check the detailed README.md (comprehensive methodology)
2. Review example code in `src/generation/pipeline.py`
3. Test with `scripts/smoke_test.py` to isolate problems
4. Check GPU/CUDA setup if generation is failing

---

## Next Document to Create

After collecting initial data, create:
- `notebooks/01_data_exploration.ipynb` - Analyze your dataset
- `src/data_processing/segment_motifs.py` - Preprocessing pipeline
- `data/annotations/annotation_guide.md` - Detailed annotation instructions

---

**Start with Step 1 (smoke test) today to validate your setup!**

