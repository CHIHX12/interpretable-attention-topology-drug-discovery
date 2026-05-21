# GHSR transfer learning
# Transfer Learning from BindingDB to GHSR

## 📋 

use BindingDB training DrugBAN_BiLSTM model GHSR datarowtransfer learningTransfer Learningfine-tuning

**transfer learning**
- ✅ data
- ✅ training
- ✅ 
- ✅ 

---

## 🚀 start

### step 1datasplitdone

```bash
# GHSR datasplit train/val/test
datasets/GPCR_resarch/random/
├── train.csv  (1077 samples, 70%)
├── val.csv    (153 samples, 10%)
└── test.csv   (309 samples, 20%)

# uselayersamplingactive/activeratio ~50/50
```

split
```bash
python3 split_ghsr_data.py --stratify --seed 42
```

---

### step 2transfer learningtraining

```bash
# use
bash run_ghsr_transfer_learning.sh

# use main.py
python3 main.py \
    --cfg configs/DrugBAN_BiLSTM_GHSR_TransferLearning.yaml \
    --data GPCR_resarch \
    --split random
```

---

## ⚙️ config

### transfer learningconfigfile
`configs/DrugBAN_BiLSTM_GHSR_TransferLearning.yaml`

**keyparameter：**
```yaml
SOLVER:
 PRETRAINED_MODEL: "result/DrugBAN_BiLSTM/best_model_epoch_94.pth" # trainingmodel
 LR: 1e-5 # learning ratetrainingweight
 MAX_EPOCH: 50 # fine-tuning epoch
 BATCH_SIZE: 32 # data

PROTEIN:
 USE_BILSTM: True # ✅ trainingmodel
  LSTM_HIDDEN_DIM: 256
  LSTM_NUM_LAYERS: 2

DRUG:
 USE_BILSTM: False # ✅ DrugBAN_BiLSTM use GCN 
```

**⚠️ important**
- parametertrainingmodel
- learning rate epoch 
- modelmatch

---

## 📊 result

### trainingoutput
```
result/DrugBAN_BiLSTM_GHSR_TransferLearning/
├── best_model_epoch_XX.pth # model
├── model_epoch_50.pth          # finalmodel
├── result.txt # 
└── selfies_vocab.pkl # Vocabularyuse BiLSTM
```

### 
 GHSR test set
- **AUROC**: > 0.80transfer learningvs ~0.75training
- **AUPRC**: > 0.75
- **Accuracy**: > 0.70

---

## 🔬 analysis

### 1. predictionattentionextract

usetrainingmodelextractattentionscore

```bash
python3 batch_predict_ghsr.py \
    --config configs/DrugBAN_BiLSTM_GHSR_TransferLearning.yaml \
    --model_path result/DrugBAN_BiLSTM_GHSR_TransferLearning/best_model_epoch_XX.pth \
    --output_dir datasets/GPCR_resarch/attention_results_TransferLearning
```

### 2. consensusanalysis

importantbinding site

```bash
python3 consensus_analysis_ghsr.py \
    --attention_file datasets/GPCR_resarch/attention_results_TransferLearning/GHSR_attention_scores.npz \
    --output_dir datasets/GPCR_resarch/consensus_results_TransferLearning
```

### 3. structurevalidate

predictionbinding site PDB pocket

```bash
bash run_ghsr_validation.sh \
    --model result/DrugBAN_BiLSTM_GHSR_TransferLearning/best_model_epoch_XX.pth \
    --config configs/DrugBAN_BiLSTM_GHSR_TransferLearning.yaml
```

### 4. generate PyMOL visualization

```bash
python3 package_ghsr_pymol.py
```

---

## 🆚 comparison

### comparisontransfer learning vs. training

1. **training GHSR**
```bash
# configtrainingmodel
python3 main.py \
    --cfg configs/DrugBAN_BiLSTM.yaml \
    --data GPCR_resarch \
    --split random
```

2. **comparison**
- 
- finalAUROC, AUPRC, Accuracy
- test set
- attentionscore

### usecomparison

comparisonmodel

```bash
bash compare_ghsr_models.sh
# modelpath
```

---

## 🔧 

### learning rate

modellearning rate

```yaml
SOLVER:
 LR: 5e-6 # learning ratefine-tuning
 # 
 LR: 5e-5 # learning rate
```

### training epoch

```yaml
SOLVER:
 MAX_EPOCH: 30 # epoch
 # 
 MAX_EPOCH: 100 # epoch
```

### useEarly Stopping

trainingautosavevalidation setmodel`best_model_epoch_XX.pth`

---

## 📚 

### transfer learning

 `main.py` autorow

```python
# trainingmodel
if cfg.SOLVER.PRETRAINED_MODEL and os.path.exists(cfg.SOLVER.PRETRAINED_MODEL):
    checkpoint = torch.load(cfg.SOLVER.PRETRAINED_MODEL, map_location=device)
    model.load_state_dict(checkpoint, strict=True)
    print("✅ Pretrained model loaded successfully!")
```

### datasplit

uselayersamplingStratified Split
- eachsplitactive/activeratio
- class
- Random seed = 42

---

## ❓ 

### Q1: trainingmodel
**A:** BindingDB trainingmodel
```bash
ls result/DrugBAN_BiLSTM/best_model_epoch_94.pth
```

### Q2: modelmatcherror
**A:** configfileparametertrainingmodel
- `PROTEIN.USE_BILSTM`
- `PROTEIN.LSTM_HIDDEN_DIM`
- `DRUG.USE_BILSTM`
- 

### Q3: training
**A:** batch size learning rate
```yaml
SOLVER:
 BATCH_SIZE: 64 # 
 LR: 5e-6 # learning rate epoch
```

### Q4: datasplit
**A:** seed split
```bash
python3 split_ghsr_data.py --stratify --seed 123
```

---

## 📖 data

### file
- `BiLSTM_README.md` - BiLSTM 
- `CONFIG_GUIDE.md` - configfile
- `CONSENSUS_ANALYSIS_README.md` - consensusanalysis

### createfile
```
split_ghsr_data.py # datasplit
configs/DrugBAN_BiLSTM_GHSR_TransferLearning.yaml    # transfer learningconfig
run_ghsr_transfer_learning.sh # training
datasets/GPCR_resarch/random/ # splitdata
```

---

## 🎯 

```
1. datasplit ✅
   ↓
2. transfer learningtraining ← 
   ↓
3. predictionattentionextract
   ↓
4. consensusanalysis
   ↓
5. structurevalidate
   ↓
6. PyMOL visualization
   ↓
7. 
```

---

**update**: 2024-12-25
****: Claude Code
****: GHSR transfer learning
