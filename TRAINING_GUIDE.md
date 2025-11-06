# StyleGAN3 Training Guide for Greek Motifs

## ✅ Prerequisites Complete

- [x] StyleGAN3 installed and working
- [x] Greek motifs preprocessed (520+ images)
- [x] Data organized by region (11 regions)
- [x] Metadata extracted

---

## 🚀 Step-by-Step Training Instructions

### Step 1: Prepare Dataset for StyleGAN3

Run this in your **CulturalGaN** directory:

```bash
cd "OneDrive - ΠΑΝΕΠΙΣΤΗΜΙΟ ΘΕΣΣΑΛΙΑΣ\Επιφάνεια εργασίας\Github\CulturalGaN"
python scripts/prepare_stylegan_dataset.py
```

This will create:
- `data/processed/dataset.json` - StyleGAN3 dataset file
- `data/processed/region_labels.json` - Region-to-label mapping

---

### Step 2: Verify Dataset Structure

Your `data/processed/` should now contain:

```
data/processed/
├── dataset.json          ← Required by StyleGAN3
├── region_labels.json    ← Region mapping
├── metadata.csv          ← Your preprocessing results
├── Aegean_Islands/       ← Image directories
├── Cyclades/
├── Dodecanese/
├── Epirus/
├── Greece/
├── Lesvos/
├── North_Aegean/
├── Rhodes/
├── Thessaly/
├── Thrace/
└── Turkey/
```

---

### Step 3: Train StyleGAN3

Navigate to your **StyleGAN3** directory and run:

```bash
cd ..\stylegan3

# Basic training command
python train.py \
  --outdir=../CulturalGaN/models/checkpoints/greek-motifs-run1 \
  --cfg=stylegan3-t \
  --data=../CulturalGaN/data/processed \
  --gpus=1 \
  --batch=16 \
  --gamma=8.2 \
  --cond=1 \
  --mirror=1 \
  --snap=10 \
  --metrics=fid50k_full
```

**PowerShell syntax** (Windows):

```powershell
python train.py `
  --outdir=..\CulturalGaN\models\checkpoints\greek-motifs-run1 `
  --cfg=stylegan3-t `
  --data=..\CulturalGaN\data\processed `
  --gpus=1 `
  --batch=16 `
  --gamma=8.2 `
  --cond=1 `
  --mirror=1 `
  --snap=10 `
  --metrics=fid50k_full
```

---

### Step 4: Training Parameters Explained

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| `--cfg` | `stylegan3-t` | StyleGAN3 translation-equivariant (good for patterns) |
| `--data` | `../CulturalGaN/data/processed` | Your processed dataset |
| `--gpus` | `1` | Number of GPUs (adjust if you have more) |
| `--batch` | `16` | Batch size (reduce if out of memory) |
| `--gamma` | `8.2` | R1 regularization (higher = more stable) |
| `--cond` | `1` | Enable conditional generation (region labels) |
| `--mirror` | `1` | Horizontal mirroring augmentation |
| `--snap` | `10` | Save checkpoint every 10 ticks |
| `--metrics` | `fid50k_full` | Calculate FID metric |

---

### Step 5: Monitor Training

**Option A: Watch the terminal**
- Training progress will be printed to console
- Shows current tick, kimg, time, FID scores

**Option B: TensorBoard** (Recommended)

In a **new terminal**:

```bash
cd "OneDrive - ΠΑΝΕΠΙΣΤΗΜΙΟ ΘΕΣΣΑΛΙΑΣ\Επιφάνεια εργασίας\Github\CulturalGaN"
tensorboard --logdir=models/checkpoints
```

Then open: http://localhost:6006

---

### Step 6: Training Time Estimates

With **RTX 3060 (8GB VRAM)**:
- **Batch size 16**: ~520 images, ~32 batches per tick
- **Expected**: 1-2 minutes per tick
- **Good results**: ~1000 ticks (25-50 hours)
- **High quality**: ~5000 ticks (5-10 days)

**Tips to speed up**:
- Reduce `--batch` if out of memory (8 or 4)
- Disable metrics initially: remove `--metrics` flag
- Use mixed precision (automatic in StyleGAN3)

---

### Step 7: Generate Test Images During Training

**After ~100 ticks**, test your model:

```bash
# In stylegan3 directory
python gen_images.py \
  --network=../CulturalGaN/models/checkpoints/greek-motifs-run1/network-snapshot-000100.pkl \
  --seeds=0-15 \
  --outdir=../CulturalGaN/outputs/test-generation \
  --class=0
```

Change `--class` to generate from different regions:
- `--class=0` → Aegean_Islands
- `--class=1` → Cyclades
- `--class=2` → Dodecanese
- etc. (see `region_labels.json`)

---

### Step 8: Resume Training (if interrupted)

```bash
python train.py \
  --outdir=../CulturalGaN/models/checkpoints/greek-motifs-run1 \
  --cfg=stylegan3-t \
  --data=../CulturalGaN/data/processed \
  --resume=../CulturalGaN/models/checkpoints/greek-motifs-run1/network-snapshot-latest.pkl \
  --gpus=1 \
  --batch=16 \
  --cond=1
```

---

## 🎯 Training Configurations

### Quick Test (30 min - 1 hour)
```bash
python train.py \
  --cfg=stylegan3-t \
  --data=../CulturalGaN/data/processed \
  --outdir=../CulturalGaN/models/checkpoints/test-run \
  --gpus=1 --batch=8 --gamma=8.2 --cond=1 --kimg=100
```

### Production Training (5-10 days)
```bash
python train.py \
  --cfg=stylegan3-t \
  --data=../CulturalGaN/data/processed \
  --outdir=../CulturalGaN/models/checkpoints/production-run \
  --gpus=1 --batch=16 --gamma=8.2 --cond=1 --mirror=1 \
  --snap=50 --metrics=fid50k_full --kimg=25000
```

---

## 📊 What to Expect

### Early Training (0-500 ticks)
- Images will be blurry/abstract
- Colors start to appear
- Basic shapes emerge

### Mid Training (500-2000 ticks)
- Recognizable patterns
- Better symmetry
- Color palettes improve

### Late Training (2000-5000+ ticks)
- High-quality motifs
- Sharp details
- Authentic-looking patterns
- Regional characteristics visible

---

## ⚠️ Troubleshooting

### Out of Memory
```bash
# Reduce batch size
python train.py ... --batch=8  # or even --batch=4
```

### Training is slow
```bash
# Disable metrics during training
python train.py ... # remove --metrics flag

# Or use lighter metric
python train.py ... --metrics=is50k
```

### "Dataset not found"
- Check that `data/processed/dataset.json` exists
- Run `python scripts/prepare_stylegan_dataset.py`
- Verify path in training command

### CUDA out of memory
- Close other GPU applications
- Reduce `--batch` to 4 or 8
- Use smaller resolution (modify dataset)

---

## 📁 Output Files

Training creates in `models/checkpoints/greek-motifs-run1/`:

- `network-snapshot-XXXXXX.pkl` - Model checkpoints
- `training_options.json` - Training configuration
- `log.txt` - Training log
- `metric-fid50k_full.jsonl` - FID scores over time

---

## 🎨 After Training: Generate Motifs

Once you have a trained model:

```bash
# Generate 50 random Greek motifs
python gen_images.py \
  --network=../CulturalGaN/models/checkpoints/greek-motifs-run1/network-snapshot-005000.pkl \
  --seeds=0-49 \
  --outdir=../CulturalGaN/outputs/generated-motifs \
  --trunc=0.7

# Generate region-specific motifs
# Greece region (class from region_labels.json)
python gen_images.py \
  --network=../CulturalGaN/models/checkpoints/greek-motifs-run1/network-snapshot-005000.pkl \
  --seeds=0-20 \
  --class=4 \
  --outdir=../CulturalGaN/outputs/greece-motifs \
  --trunc=0.7
```

---

## 📝 Next Steps After Training

1. **Evaluate Quality**
   - Visual inspection of generated motifs
   - Compare to original dataset
   - Check FID scores

2. **Generate Variations**
   - Try different `--trunc` values (0.5 to 1.0)
   - Test different `--class` labels (regions)
   - Create interpolations

3. **Expert Evaluation**
   - Show to Greek cultural experts
   - Assess authenticity preservation
   - Document findings

4. **Research Documentation**
   - Document training process
   - Record metrics
   - Create case studies
   - Write paper

---

## 🔗 Useful Commands Reference

```bash
# Prepare dataset
cd CulturalGaN
python scripts/prepare_stylegan_dataset.py

# Start training
cd ..\stylegan3
python train.py --cfg=stylegan3-t --data=../CulturalGaN/data/processed --outdir=../CulturalGaN/models/checkpoints/run1 --gpus=1 --batch=16 --cond=1

# Monitor with TensorBoard
cd ..\CulturalGaN
tensorboard --logdir=models/checkpoints

# Generate images
cd ..\stylegan3
python gen_images.py --network=../CulturalGaN/models/checkpoints/run1/network-snapshot-latest.pkl --seeds=0-15 --outdir=../CulturalGaN/outputs/test

# Resume training
python train.py --resume=../CulturalGaN/models/checkpoints/run1/network-snapshot-latest.pkl --gpus=1 --batch=16
```

---

## ✅ Current Status

**Phase 1: Data Preparation** ✅ COMPLETE
- [x] Dataset collected (520+ images)
- [x] Images preprocessed
- [x] Features extracted
- [x] Metadata created

**Phase 2: Model Setup** ✅ COMPLETE
- [x] StyleGAN3 installed
- [x] Dataset preparation script ready
- [x] Training configuration ready

**Phase 3: Training** ⬅️ **YOU ARE HERE**
- [ ] Run dataset preparation script
- [ ] Start training
- [ ] Monitor progress
- [ ] Wait for convergence

**Phase 4: Evaluation** (After Training)
- [ ] Generate test motifs
- [ ] Calculate metrics
- [ ] Expert review
- [ ] Research paper

---

**Ready to train!** Run the preparation script, then start training with the commands above.

Good luck! 🏛️🇬🇷

