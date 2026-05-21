# PyMOL Visualization Guide for Class-Specific Attention

## 📋 

### ✅ Q1: 
**A**: `result/class_attention_analysis/`version
versionanalysis `analyze_class_attention_difference_en.py`
generaterow

```bash
python3 analyze_class_attention_difference_en.py \
    --config configs/DrugBAN_BiLSTM_GHSR_TransferLearning.yaml \
    --model_path result/DrugBAN_BiLSTM_GHSR_Seed42/best_model_epoch_44.pth \
    --data_file datasets/GPCR_resarch/GHSR_training_data.csv \
    --output_dir result/class_attention_analysis_en \
    --device cuda
```

### ✅ Q2: amino acidactualamino acid
**A**: **PyMOL useall PDB actualresidue**


- `resi 287` → PDB structure 287 residueS287
- `resi 286` → PDB structure 286 residueF286

**convert**

 PyMOL `resi 287` S287
validate

### ✅ Q3: Class 0 Class 1 
**A**: **done**

generate PyMOL 
1. `pymol_class0_inactive_preferred.pml` - Class 0activeresidue
2. `pymol_class1_active_preferred.pml` - Class 1activeresidue

color
- **Red**: Top 10 residue
- **Orange**: intermediate 11-20 
- **Yellow**: 21-30 
- **Cyan sticks**: pocket
- **Magenta**: pocketoverlappartial

---

## 🎨 usemethod

### 1. Class 0activevisualization

```bash
pymol pymol_class0_inactive_preferred.pml
```

****:
- ****: S287, F286, P278, F147 active
- ****: pocket
- ****: F286, P278 pocket

**key**: 10 // 10 pocket
→ activepocket

### 2. Class 1activevisualization

```bash
pymol pymol_class1_active_preferred.pml
```

****:
- ****: E124, S125, V122 active
- ****: pocket
- ****: C126pocketoverlap

**key**: 1 residueC126pocket
→ activepocketregion

---

## 📊 residue

### PDB actual

| PyMOL | | validate |
|---------------|------|-----------|
| resi 287 | PDB 8JSR 287 residueS | ✅ |
| resi 286 | PDB 8JSR 286 residueF | ✅ |
| resi 278 | PDB 8JSR 278 residueP | ✅ |
| resi 124 | PDB 8JSR 124 residueE | ✅ |

### convert

 PyMOL use PDB data

```python
Dataset_Position = PDB_Residue - 1

# 
# PDB 287 → Dataset Position 286
# PDB 286 → Dataset Position 285
```

** PyMOL visualizationconvert**

---

## 🔍 Top residuecolumnPDB 

### Class 0 (Inactive-preferred) Top 10

| ranking | PDB residue | difference | P-value | pocket |
|------|---------|------|---------|------------|
| 1 | S287 | -0.773 | 1.6e-19 | ❌ |
| 2 | F286 | -0.685 | 3.7e-23 | ✅ **** |
| 3 | P278 | -0.640 | 1.5e-15 | ✅ **** |
| 4 | F147 | -0.640 | 3.5e-36 | ❌ |
| 5 | C116 | -0.623 | 5.8e-29 | ❌ |
| 6 | Q302 | -0.594 | 2.0e-12 | ❌ |
| 7 | P41 | -0.550 | 1.5e-22 | ❌ |
| 8 | F38 | -0.547 | 2.7e-23 | ❌ |
| 9 | P39 | -0.531 | 1.3e-24 | ❌ |
| 10 | P292 | -0.523 | 9.8e-15 | ❌ |

**pocketoverlap**: Top 30 10 pocket

### Class 1 (Active-preferred) Top 10

| ranking | PDB residue | difference | P-value | pocket |
|------|---------|------|---------|------------|
| 1 | E124 | +0.244 | 4.7e-08 | ❌ |
| 2 | S125 | +0.216 | 1.1e-09 | ❌ |
| 3 | V122 | +0.080 | 1.6e-03 | ❌ |
| 4 | C198 | +0.065 | 1.6e-15 | ❌ |
| 5 | P200 | +0.049 | 1.1e-02 | ❌ |
| 6 | Q299 | +0.043 | 2.5e-09 | ❌ |
| 7 | Y232 | +0.028 | 1.4e-14 | ❌ |
| 8 | C126 | +0.028 | 1.0e-21 | ✅ **** |
| 9 | V230 | +0.024 | 4.8e-25 | ❌ |
| 10 | V160 | ~0.000 | 1.2e-60 | ❌ |

**pocketoverlap**: 1 C126pocket

---

## 🎯 key

### 1: activepocket

****:
- Class 0 Top 30 residue10 pocket67%
- F286 P278p < 1e-15

****:
- active****pocket
- ****activationreceptor

### 2: activepocketregion

****:
- Class 1 Top 10 1 pocket
- E124, S125pocket

****:
- activeusemode
- pocketresidue

---

## 📸 

### PyMOL 

1. ****
```bash
pymol pymol_class0_inactive_preferred.pml
```

2. ****
```python
# PyMOL console 
orient class0_tier1
zoom class0_tier1, 5
```

3. ****
```python
# 
ray 2400, 2400
png class0_visualization_high_res.png, dpi=300
```

4. ** Class 1**
```bash
pymol pymol_class1_active_preferred.pml
# ... step
```

---

## 📝 

### Figure Caption 

**Figure X: Class-specific attention reveals distinct binding preferences**

**(A)** PyMOL visualization of Class 0 (inactive) preferred residues mapped onto
GHSR crystal structure (PDB: 8JSR). Spheres represent residues with significantly
higher attention for inactive compounds, colored by preference strength: red
(highest, top 10), orange (moderate, rank 11-20), yellow (lower, rank 21-30).
Cyan sticks show the experimentally-determined binding pocket. Magenta highlights
overlap between attention-predicted residues and the experimental pocket.

**(B)** PyMOL visualization of Class 1 (active) preferred residues. Note the
limited overlap with the experimental pocket (only C126), suggesting active
compounds utilize different binding modes or engage additional interaction sites.

### Results Section 

> "Class-specific attention analysis identified 130 residues within the crystallographic
> structure (PDB 8JSR, residues 37-338) showing preferential attention for inactive
> compounds. Notably, 10 of these residues overlapped with the experimentally-determined
> binding pocket, including F286 (p=3.7e-23) and P278 (p=1.5e-15). In contrast, only
> 1 of 31 active-preferred residues (C126) overlapped with the experimental pocket,
> with the majority clustering near the pocket entrance (E124, S125). This divergence
> suggests that productive receptor activation requires specific engagement patterns
> beyond simple pocket occupancy."

---

## 🔧 use

### color

color `.pml` file

```python
# 
color red, class0_tier1
# 
color deepred, class0_tier1

# colorred, orange, yellow, green, blue, purple, pink, etc.
```

### display Top 5

```python
# PyMOL 
hide spheres, class0_tier2
hide spheres, class0_tier3
# Tier 1 (Top 10)
```

### displayclasscomparisonmode

```python
# structure
load datasets/GPCR_resarch/GSHR_PDB/8JSR_R.pdb

# Class 0 ()
select class0_top10, chain R and resi 287+286+278+147+116+302+41+38+39+292
color blue, class0_top10
show spheres, class0_top10

# Class 1 ()
select class1_top10, chain R and resi 124+125+122+198+200+299+232+126+230+160
color red, class1_top10
show spheres, class1_top10

# pocket
select exp_pocket, chain R and resi 37+40+99+102+103+106+110+123+126+127+278+282+285+286+289
color yellow, exp_pocket
show sticks, exp_pocket
```

---

## 📁 generatefile

```
pymol_class0_inactive_preferred.pml # Class 0 visualization
pymol_class1_active_preferred.pml # Class 1 visualization
result/pymol_visualization_summary.txt # file
```

---

## ✅ check

use

- [ ] confirm PyMOL residue PDB 
- [ ] PyMOL checkvisualization
- [ ] ray 2400, 2400
- [ ] validateresidue
- [ ] confirmcolor
- [ ] //

---

## 🆘 

### Q: 
A: PDB 37-338 Dataset Position1-523

### Q: confirmresidue
A: PyMOL displayresidue

### Q: size
A: PyMOL `set sphere_scale, 1.0, class0_tier1`

### Q: 
A: use`orient`, `zoom`, `center` 

---

**filecreate**: 2024-12-25
**PDB structure**: 8JSR Chain R (Residues 37-338)
****: use PDB actualconvert
