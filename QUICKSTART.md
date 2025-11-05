# CulturalGaN - Quick Start Guide

**Project:** AI-Powered Greek Motif Adaptation for Contemporary Fashion  
**Status:** Phase 1 - Dataset Preparation

---

## ⚡ 3-Step Quick Start

### Step 1: Verify Your Setup (5 minutes)

Activate your virtual environment and test the pipeline:

```powershell
# Activate environment
.\venv\Scripts\Activate.ps1

# Run smoke test
python scripts\smoke_test.py
```

**What this does:**
- Downloads Stable Diffusion model (~4GB, first time only)
- Generates a test fashion design
- Saves output to `outputs/generated_designs/smoke_output.png`

**Expected output:**
```
Input:  outputs\generated_designs\smoke_input.png
Output: outputs\generated_designs\smoke_output.png
Size:   (512, 512)
```

---

### Step 2: Collect Your First Motifs (Today)

**Goal:** Get 10-20 Greek motif images as a starting point

**Where to find them:**

1. **Wikimedia Commons** (easiest, public domain)
   - Search: "Greek embroidery"
   - Search: "Greek pottery patterns"
   - Search: "Byzantine motifs"
   - Download high-resolution images

2. **Museum Collections** (requires some digging)
   - Benaki Museum Digital Collection
   - Metropolitan Museum of Art (search "Greek")

3. **Books & PDFs** (scan or screenshot)
   - Public domain books on Greek folk art
   - Google Books preview images

**Save to:** `data/raw/` with descriptive names
- Example: `crete_floral_rose_001.png`

---

### Step 3: Test Generation with Your Motif (10 minutes)

Once you have even ONE motif image:

```bash
python scripts\smoke_test.py --input "data/raw/your_motif.png" --output "outputs/test_dress.png" --garment "dress" --adapt 0.5
```

**Parameters you can experiment with:**
- `--garment`: "dress", "blouse", "scarf", "jacket"
- `--adapt`: 0.0 (literal) to 1.0 (abstract), try 0.3, 0.5, 0.7
- `--color`: "original", "modernized", "monochrome"

**Try different combinations:**
```bash
# More literal adaptation
python scripts\smoke_test.py --input "data/raw/motif.png" --adapt 0.2 --garment "blouse"

# More abstract
python scripts\smoke_test.py --input "data/raw/motif.png" --adapt 0.8 --garment "evening dress"

# Different color strategy
python scripts\smoke_test.py --input "data/raw/motif.png" --color "monochrome" --garment "scarf"
```

---

## 📊 View Progress

Open the data exploration notebook:

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

This will show:
- How many motifs you've collected
- Dataset statistics
- Image quality metrics
- What's missing

---

## 🎯 This Week's Goals (Phase 1, Week 1)

- [ ] Verify setup works (smoke test passes)
- [ ] Collect 50-100 Greek motif images
- [ ] Organize into folders by region/type
- [ ] Test generation with 5-10 different motifs
- [ ] Document what works and what doesn't

---

## 📁 File Organization

```
data/raw/
├── crete/
│   ├── floral/
│   │   ├── crete_floral_rose_001.png
│   │   └── crete_floral_carnation_002.png
│   └── geometric/
│       └── crete_geometric_meander_001.png
├── peloponnese/
│   └── ...
└── islands/
    └── ...
```

**Naming convention:**
```
{region}_{type}_{description}_{number}.png
```

Examples:
- `crete_floral_rose_border_001.png`
- `macedonia_geometric_cross_stitch_002.png`
- `islands_zoomorphic_fish_003.png`

---

## 🔍 Common Issues

### Issue: "RuntimeError: CUDA out of memory"
**Solution:** The pipeline automatically falls back to CPU. It will be slower (2-5 min per image) but will work.

### Issue: Model download fails
**Solution:** 
```python
# Set Hugging Face cache directory
import os
os.environ['HF_HOME'] = 'C:/hf_cache'
```

### Issue: "No module named 'src'"
**Solution:** Make sure you're running from the project root directory
```bash
cd path\to\CulturalGaN
```

### Issue: Images look distorted
**Solution:** 
- Try different adaptation levels (0.3-0.7 usually works best)
- Ensure input image is at least 512x512px
- Try simpler motifs first (geometric patterns work better than complex scenes)

---

## 🎓 Learning Resources

### Understanding Stable Diffusion
- It's an image-to-image model that transforms your motif into a fashion design
- Lower `--adapt` = stays closer to original
- Higher `--adapt` = more creative interpretation

### Greek Motif Research
See detailed sources in `README.md` section 2.1 and Appendix B

---

## ✅ Success Indicators

After your first session, you should have:
1. ✅ Smoke test runs successfully
2. ✅ 10-50 Greek motif images collected
3. ✅ At least 3-5 test generations created
4. ✅ Observations about what works/doesn't work

---

## 📞 Next Steps

Once you have 50-100 motifs:
1. Read `NEXT_STEPS.md` for Week 2 tasks
2. Start annotating your motifs (use `data/annotations/template.json`)
3. Begin preprocessing and segmentation
4. Move toward the 500-motif minimum for training

---

## 💡 Pro Tips

1. **Start with high-quality images** - Better input = better output
2. **Test early and often** - Don't wait to have 500 motifs to test
3. **Document everything** - Keep notes on what works
4. **Diversity matters** - Collect from different regions and time periods
5. **Simple first** - Geometric patterns are easier than complex scenes

---

**Ready to begin? Run the smoke test now!** 🚀

```bash
python scripts\smoke_test.py
```

