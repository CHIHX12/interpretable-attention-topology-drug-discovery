# DrugBAN with BiLSTM Protein Encoder

## 

version DrugBAN **BiLSTM protein**optionversion

### 

1. **models.py**
 - `ProteinBiLSTM` 121-176row
 - `DrugBAN.__init__` switch CNN/BiLSTM60-88row
 - use `USE_BILSTM` configoption

2. **configs.py**
 - BiLSTM configparameter
   - `USE_BILSTM`: True/False switch
 - `LSTM_HIDDEN_DIM`: BiLSTM layerdimension
 - `LSTM_NUM_LAYERS`: LSTM layer
 - `LSTM_DROPOUT`: Dropout 
 - `MAX_PROTEIN_LENGTH`: maxproteinsequencelength500 12receptor

3. **configs/DrugBAN_BiLSTM.yaml**
 - use BiLSTM configfile

---

## usemethod

### method 1use BiLSTM configfile

```bash
# use BiLSTM training
python main.py --cfg "configs/DrugBAN_BiLSTM.yaml" --data bindingdb --split random
```

### method 2：useoriginal CNN

```bash
# use CNN default
python main.py --cfg "configs/DrugBAN.yaml" --data bindingdb --split random
```

### method 3config

create YAML file

```yaml
PROTEIN:
  USE_BILSTM: True          # enable BiLSTM
 LSTM_HIDDEN_DIM: 256 # layerdimensionoutput = 256*2=512
 LSTM_NUM_LAYERS: 2 # LSTM layer
  LSTM_DROPOUT: 0.2         # Dropout
  EMBEDDING_DIM: 128        # amino acid Embedding dimension
  MAX_PROTEIN_LENGTH: 500   # maxsequencelength
  PADDING: True
```

---

## BiLSTM vs CNN

| | CNN | BiLSTMversion |
|------|-------------|-------------------|
| sequence | feature | bidirectional |
| distance | kernel size | LSTM |
| length | padding | Pack/Pad processing |
| training | row | |
| parameter | | |
| sequencelength | < 1000 | < 3000 |

---

## difference

### versionCNN

```
Protein Sequence
    ↓
Embedding (128 dim)
    ↓
Conv1D (kernel=3) → BatchNorm → ReLU
    ↓
Conv1D (kernel=6) → BatchNorm → ReLU
    ↓
Conv1D (kernel=9) → BatchNorm → ReLU
    ↓
Output: [batch, seq_len, 128]
```

### BiLSTM version

```
Protein Sequence
    ↓
Embedding (128 dim)
    ↓
Pack Sequence (length)
    ↓
BiLSTM (2 layers, hidden=256)
  ├─ Forward LSTM
  └─ Backward LSTM
    ↓
Unpack Sequence
    ↓
Output: [batch, seq_len, 512]  # 256*2
```

---

## trainingdata

dockingresultdata

```python
# dataformat
dataset.csv:
- smiles: CC(C)...
- Protein: ALLT...
- residue_numbers: [307,308,310,...]
- Y: 1 0
- cnnscore: 0.823
```

training

```bash
# datarow
python main.py \
  --cfg "configs/DrugBAN_BiLSTM.yaml" \
  --data your_dataset \
  --split random
```

---

## sequencelengthset

### 12 receptorset

actualsequencelengthanalysispocketregion30Å distance

```
receptorsequencelength
5EK0: 228 aa   (Ion channel)
5MZJ: 244 aa   (GPCR)
5L2S: 245 aa   (GPCR)
2ZV2: 249 aa   (Kinase)
1T7R: 253 aa   (Nuclear receptor)
4AG8: 289 aa   (Kinase)
4R06: 308 aa   (Kinase)
4YAY: 310 aa   (GPCR)
6IIU: 315 aa   (GPCR)
1ERR: 333 aa   (Nuclear receptor)
4F8H: 378 aa   (Ion channel)
6D6T: 497 aa   (Ion channel)

: 228 aa
: 497 aa
average: 304 aa
```

**setMAX_PROTEIN_LENGTH = 500**


- ✅ 12 receptor 497 aa
- ✅ Padding minaverage 39% padding
- ✅ BiLSTM computepack_padded_sequence autoprocessingactuallength
- ✅ BAN attention

---

## attentionanalysis

BiLSTM version BAN attentionanalysisdrug-protein

```python
# attentionweight
model.eval()
v_d, v_p, score, attention = model(drug, protein, mode="eval")

# attention shape: [batch, drug_atoms, protein_residues]
# residue_numbers visualizationresidueimportant
```

---

## 

- ✅ drugMolecularGCN
- ✅ BAN attentionlayer
- ✅ MLP classification
- ✅ Domain Adaptation
- ✅ training
- ✅ dataload

**proteinswitch CNN ↔ BiLSTM**
