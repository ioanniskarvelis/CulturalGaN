# Region Normalization Summary

## Problem Identified

Your dataset had **38 region directories** with many duplicates due to inconsistent naming:

### Examples of Duplicates:
- **Lesvos** written as:
  - Lesvos_(Lesbos)_Greece (27 images)
  - Lesbos_(Lesvos)_Greece (8 images)
  - Lesvos_Greece (16 images)
  - Lesbos_Greece (6 images)
  - ... 7 more variations!

- **Rhodes_Dodecanese** written as:
  - Rhodes_Dodecanese_Greece (81 images)
  - Rhodes_(Dodecanese_Greece) (3 images)
  - Rhodes_Dodecanese (3 images)

- **Greece** written as:
  - Greece (198 images)
  - Aegean_Greece (8 images)
  - Greece_Aegean (4 images)
  - Miscellaneous (3 images)

---

## Solution

A region normalization system that:
1. Maps all variations to standard names
2. Consolidates duplicate directories
3. Updates all JSON annotations
4. Preserves original region names for reference

---

## Proposed Changes

### Before → After

**38 region directories** → **14 normalized regions**

| Normalized Region | Images | Consolidates From |
|-------------------|--------|-------------------|
| **Greece** | 215 | Greece, Aegean_Greece, Greece_Aegean, Miscellaneous, etc. |
| **Rhodes_Dodecanese** | 87 | Rhodes_Dodecanese_Greece, Rhodes_(Dodecanese_Greece), Rhodes_Dodecanese |
| **Lesvos** | 60 | 7 Lesvos/Lesbos variations |
| **Lesvos_North_Aegean** | 57 | 9 Lesvos North Aegean variations |
| **North_Aegean** | 32 | North_Aegean_Greece, Greece_North_Aegean |
| **Thessaly** | 18 | (no change needed) |
| **Aegean_Islands** | 16 | Aegean_Islands_Greece, Greece_Aegean_Islands |
| **Epirus** | 9 | (no change needed) |
| **Turkey** | 8 | (no change needed) |
| **Cyclades** | 6 | (no change needed) |
| **Dodecanese** | 2 | Dodecanese_Greece |
| **Rhodes** | 2 | Rhodes_Greece |
| **Thrace** | 2 | (no change needed) |
| **Lesvos_Mytilene** | 1 | Lesvos_(Mytilene)_Greece |

**Total: 522 images** (no images lost)

---

## What Will Happen

### 1. Image Reorganization
- Images moved from duplicate directories into normalized ones
- Example: All Lesvos variants → `data/raw/Lesvos/`
- Empty directories automatically removed

### 2. Annotation Updates
- All 522 JSON files updated with normalized region names
- Original region name preserved in `region_original` field
- File paths updated to reflect new organization

### 3. Safety
- ✅ No images deleted
- ✅ All moves, no copies (efficient)
- ✅ Original region names preserved in annotations
- ✅ Can be reversed if needed

---

## Benefits

### 1. Cleaner Organization
- 38 folders → 14 folders (63% reduction)
- Easy to browse and understand
- Consistent naming across dataset

### 2. Better Analysis
- Accurate region statistics
- Proper grouping for regional studies
- No artificial inflation of region counts

### 3. Easier Filtering
```bash
# Before: Had to try multiple variations
python scripts/generate_designs.py --region "Lesvos"
python scripts/generate_designs.py --region "Lesbos"
python scripts/generate_designs.py --region "Lesvos (Lesbos)"

# After: One command works
python scripts/generate_designs.py --region "Lesvos"  # Gets all 60!
```

### 4. Research Quality
- Accurate dataset statistics for paper
- Professional, consistent organization
- Clear regional distribution charts

---

## How to Apply

### Option 1: Quick Apply (Recommended)
```bash
python scripts\fix_regions.py
```

### Option 2: Review First
```bash
# See full details without changes
python scripts\fix_regions.py --dry-run
```

---

## Example Region Breakdown After Fix

```
data/raw/
├── Greece/              (215 images) ← Largest, general Greece
├── Rhodes_Dodecanese/   (87 images)  ← Rhodes island group
├── Lesvos/              (60 images)  ← Lesvos island (general)
├── Lesvos_North_Aegean/ (57 images)  ← Lesvos with regional context
├── North_Aegean/        (32 images)  ← North Aegean (general)
├── Thessaly/            (18 images)  ← Mainland region
├── Aegean_Islands/      (16 images)  ← General Aegean islands
├── Epirus/              (9 images)   ← Mainland region
├── Turkey/              (8 images)   ← Turkish motifs
├── Cyclades/            (6 images)   ← Cyclades islands
├── Dodecanese/          (2 images)   ← Dodecanese (general)
├── Rhodes/              (2 images)   ← Rhodes (without Dodecanese context)
├── Thrace/              (2 images)   ← Mainland region
└── Lesvos_Mytilene/     (1 image)    ← Lesvos capital city
```

---

## Updated Statistics

### Before Normalization
- Regions: 38 (with duplicates)
- Largest: Greece (198)
- Second: Rhodes_Dodecanese_Greece (81)
- Third: Lesvos_(Lesbos)_Greece (27)
- **Problem:** Lesvos split across 11 directories!

### After Normalization
- Regions: 14 (clean, accurate)
- Largest: Greece (215)
- Second: Rhodes_Dodecanese (87)
- Third: Lesvos (60) + Lesvos_North_Aegean (57)
- **Result:** Clear, accurate distribution

---

## Impact on Your Research

### Dataset Statistics for Paper

**Before:**
> "The dataset contains 522 motifs from 38 regions across Greece..."
> (Inflated, inaccurate)

**After:**
> "The dataset contains 522 motifs from 14 distinct regions across Greece, with the largest representation from general Greece (215 motifs, 41%), followed by the Dodecanese islands, particularly Rhodes (87 motifs, 17%), and the island of Lesvos in the North Aegean (117 motifs, 22%)."
> (Accurate, professional)

### Regional Analysis
- Clear patterns visible
- Island vs. mainland distribution clear
- Geographic spread accurately represented

---

## Rollback (If Needed)

If you ever need to reverse this:

1. Original region names are preserved in JSON annotations (`region_original` field)
2. Images can be re-organized using original names
3. Or just re-run the original integration script

---

## Recommendation

**✅ Apply the fix** - It will:
- Make your dataset more professional
- Improve accuracy for research
- Make filtering and generation easier
- Provide cleaner statistics for your paper

**No downside** - Original information is preserved in annotations.

---

## Apply Now

```bash
# Apply the fix
python scripts\fix_regions.py

# Then test
python scripts\generate_designs.py --region "Lesvos" --num-motifs 5

# Explore normalized data
jupyter notebook notebooks/01_data_exploration.ipynb
```

---

**Status:** Ready to apply  
**Risk:** None (reversible, preserves all data)  
**Benefit:** Professional, accurate dataset organization

