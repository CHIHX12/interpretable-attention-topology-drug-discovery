# 🚀 GHSR validatestart
## Quick Start Guide for GHSR Validation

**1** | **One-minute Quick Start**

---

## ⚡ row

```bash
# directory
cd /home/cycheng/800milion_GPR/predict_affi/New_Docking/network/DrugBAN_BiLSTM

# rowvalidate5-15
bash run_ghsr_validation.sh
```

**** 🎉

---

## 📊 result

```bash
# 1. validatereport
cat datasets/GPCR_resarch/validation_results/GHSR_validation_report.txt

# 2. Top 20 consensusresidue
head -21 datasets/GPCR_resarch/consensus_results/GHSR_consensus_residues.csv

# 3. 
cat datasets/GPCR_resarch/validation_results/GHSR_validation_metrics.csv
```

---

## 🎨 PyMOL visualization

```bash
# pocket
pymol datasets/GPCR_resarch/validation_results/visualize_GHSR_pocket.pml

# prediction vs. 
pymol datasets/GPCR_resarch/validation_results/compare_prediction_vs_truth.pml

# consensusresidue
pymol datasets/GPCR_resarch/consensus_results/visualize_GHSR_consensus.pml
```

---

## 📈 

 `datasets/GPCR_resarch/` directory

```
consensus_results/
├── GHSR_consensus_heatmap.png # sequenceconsensusheatmap
└── GHSR_top_consensus_residues.png # Top 30 residue

validation_results/
├── GHSR_performance_vs_k.png # 4
└── GHSR_overlap_top30.png # predictionvsoverlap
```

---

## 🎯 key

validatereport

```
3. (F1-Score)
 Precision: XX.XX% ← prediction
 Recall: XX.XX% ← pocket
 F1-Score: X.XXXX ← 
 Enrichment: XX.XXx ← vs. 
   P-value: X.XXe-XX    ← statisticssignificance
```

**success**:
- ✅ Precision >50%, Recall >50% → successpocket
- ⚠️ Precision 30-50% → partial
- ❌ Precision <30% → 

---

## 🔧 option

### modelpath

```bash
bash run_ghsr_validation.sh \
    --model result/DrugBAN_BiLSTM_DA/best_model.pth \
    --config configs/DrugBAN_BiLSTM_DA.yaml
```

### batchsize

```bash
bash run_ghsr_validation.sh --batch-size 16
```

### use CPU GPU 

```bash
bash run_ghsr_validation.sh --device cpu
```

---

## 📝 

### Methods 

****:

```
To validate the biological relevance of our attention mechanism, we performed
binding pocket prediction using the GHSR dataset (1,539 ligands). Ground truth
was extracted from PDB 8JSR (15 residues within 4Å of Anamorelin). We computed
attention consensus across all ligands and evaluated performance using precision,
recall, F1-score, and enrichment metrics (Fisher's exact test).
```

### Results 

**actual**:

```
Our approach identified GHSR binding pocket residues with [XX]% precision and
[XX]% recall (F1=[X.XX]). Among top 30 residues, [XX] were true pocket residues,
representing [XX]-fold enrichment over random (p<[X.XXe-XX]). Key residues W103,
F120, and W214 ranked in the top 10, validating the model's interpretability.
```

---

## 🆘 

### Q1: modelfile

```bash
# model
ls -lh result/DrugBAN_BiLSTM*/best_model.pth

# use
result/DrugBAN_BiLSTM/best_model.pth # model
```

### Q2: row

- prediction: 3-10 GPU
- consensusanalysis: <1 
- validatecomparison: <1 
- ****: 5-15 

### Q3: 

```bash
# method1: batch
--batch-size 16

# method2: useCPU
--device cpu
```

### Q4: result

1. checkmodeltraining
2. confirmsequencealign
3. attention
4. activegroupanalysis

---

## 📚 

| | |
|------|------|
| `GHSR_QUICK_START.md` | file - start |
| `GHSR_VALIDATION_README.md` | use |
| `GHSR_VALIDATION_GUIDE.md` | validate |
| `GHSR_VALIDATION_SUMMARY.md` | summary |

---

## ✅ check

row
- [ ] directory
- [ ] confirmmodel`ls result/DrugBAN_BiLSTM/best_model.pth`
- [ ] confirmdata`ls datasets/GPCR_resarch/GHSR_training_data.csv`

row
- [ ] row `bash run_ghsr_validation.sh`
- [ ] done5-15

row
- [ ] validatereport
- [ ] check
- [ ] PyMOL visualization
- [ ] result

---

## 🎉 output

rowsuccess

```
================================================================================
🎉 GHSR validatedone
================================================================================

📁 outputfile:

1. attentionscore:
   datasets/GPCR_resarch/attention_results/GHSR_attention_scores.npz

2. consensusanalysis:
   datasets/GPCR_resarch/consensus_results/GHSR_consensus_residues.csv

3. validateresult:
 datasets/GPCR_resarch/validation_results/GHSR_validation_report.txt ← report

🏆 (Top-XX):
   Precision:  XX.XX%
   Recall:     XX.XX%
   F1-Score:   X.XXXX
   Enrichment: XX.XXx (vs. random)

✅ : [modelsuccess/partialsuccess/] GHSR pocket
```

---

**startvalidate** 🚀

```bash
bash run_ghsr_validation.sh
```

---

**create**: 2024-12-24
**version**: 1.0
****: 5-15 
****: ⭐ ()
