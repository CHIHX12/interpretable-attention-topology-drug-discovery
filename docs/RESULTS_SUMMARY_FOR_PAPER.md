# DrugBAN resultsummary ()

****: 202512
**data**: BindingDB (cluster split)
****: 16 

---

## 📊 (Key Findings)

### 1. model

**Top 3 model** ( AUROC sort):

| Rank | model | | AUROC | AUPRC | Accuracy | |
|------|------|-----------|--------|--------|----------|------|
| 🥇 1 | DrugBAN_BiLSTM | Drug GCN + Protein BiLSTM | **0.9641** | **0.9542** | **0.9065** | |
| 🥈 2 | DrugBAN | Drug GCN + Protein CNN | 0.9603 | 0.9485 | 0.9067 | Baseline |
| 🥉 3 | DrugBAN_DrugBiLSTM_ProteinBiLSTM_ProteinFeatures | Drug BiLSTM + Protein BiLSTM | 0.9564 | 0.9437 | 0.8975 | BiLSTM + feature |

### 2. comparison

#### 2.1 drug (Drug Encoder)

| type | average AUROC | AUROC | |
|------|-----------|-----------|------|
| **GCN (Graph)** | 0.7695 | 0.9641 | structure |
| **BiLSTM (Sequence)** | 0.7550 | 0.9564 | sequence |

#### 2.2 protein (Protein Encoder)

| type | average AUROC | AUROC | |
|------|-----------|-----------|------|
| **BiLSTM** | 0.7847 | 0.9641 | 🏆 ** CNN** |
| **CNN** | 0.7504 | 0.9603 | baseline |

****: Protein BiLSTM significant (+4.5% average AUROC)

### 3. physicochemicalfeature

| featuretype | | average AUROC | |
|---------|--------|-----------|---------|
| feature (Baseline) | 4 | 0.7920 | DrugBAN_BiLSTM (0.9641) |
| drugfeature | 2 | 0.7686 | DrugBAN_DrugBiLSTM_Features (0.9321) |
| proteinfeature | 2 | 0.7743 | DrugBAN_DrugBiLSTM_ProteinBiLSTM_ProteinFeatures (0.9564) |
| feature | 2 | 0.7926 | DrugBAN_DrugBiLSTM_BothFeatures (0.9392) |

****:
- feature baseline model
- proteinfeature > drugfeature
- featuredisplay

### 4. domain adaptation (Domain Adaptation) methodanalysis

#### 4.1 

| DA method | | average AUROC | average AUPRC | average Accuracy |
|---------|--------|-----------|-----------|--------------|
| **None (Baseline)** | 6 | **0.9476** | **0.9339** | **0.8910** |
| CDAN | 6 | 0.6079 | 0.5605 | 0.4994 |
| DANN | 2 | 0.6069 | 0.5522 | 0.4680 |
| MMD | 2 | 0.5964 | 0.5396 | 0.5121 |

#### 4.2 key

❌ **domain adaptationmethodsignificant**:
- CDAN: -35.8% AUROC
- DANN: -35.9% AUROC
- MMD: -37.0% AUROC

****:
1. datadifferencecluster split DA 
2. DA methodparameter
3. -
4. BiLSTM baseline DA regularization

#### 4.3 DA 

| | Best Baseline | Best DA | DA method | |
|--------|--------------|---------|---------|---------|
| GCN + BiLSTM | 0.9641 | 0.6410 | CDAN | -33.5% |
| GCN + CNN | 0.9603 | 0.6113 | CDAN | -36.3% |
| Drug BiLSTM + Protein BiLSTM | 0.9564 | 0.5922 | CDAN | -38.1% |
| Drug BiLSTM + CNN | 0.9536 | 0.6460 | CDAN | -32.3% |

---

## 📈 result

### result ( AUROC )

| Rank | | | feature | DAmethod | AUROC | AUPRC | Accuracy |
|------|---------|--------|------|--------|--------|--------|----------|
| 1 | DrugBAN_BiLSTM | GCN + BiLSTM | | Baseline | 0.9641 | 0.9542 | 0.9065 |
| 2 | DrugBAN | GCN + CNN | | Baseline | 0.9603 | 0.9485 | 0.9067 |
| 3 | DrugBAN_DrugBiLSTM_ProteinBiLSTM_ProteinFeatures | BiLSTM + BiLSTM | protein | Baseline | 0.9564 | 0.9437 | 0.8975 |
| 4 | DrugBAN_DrugBiLSTM | BiLSTM + CNN | | Baseline | 0.9536 | 0.9378 | 0.8922 |
| 5 | DrugBAN_DrugBiLSTM_BothFeatures | BiLSTM + CNN | feature | Baseline | 0.9392 | 0.9150 | 0.8778 |
| 6 | DrugBAN_DrugBiLSTM_Features | BiLSTM + CNN | drug | Baseline | 0.9321 | 0.9043 | 0.8672 |
| 7 | DrugBAN_DrugBiLSTM_BothFeatures_DA | BiLSTM + CNN | feature | CDAN | 0.6460 | 0.5894 | 0.4963 |
| 8 | DrugBAN_BiLSTM_DA | GCN + BiLSTM | | CDAN | 0.6410 | 0.5732 | 0.5346 |
| 9 | DrugBAN_BiLSTM_MMD | GCN + BiLSTM | | MMD | 0.6400 | 0.5943 | 0.5205 |
| 10 | DrugBAN_BiLSTM_DANN | GCN + BiLSTM | | DANN | 0.6238 | 0.5740 | 0.4716 |
| 11 | DrugBAN_DA | GCN + CNN | | CDAN | 0.6113 | 0.6007 | 0.5188 |
| 12 | DrugBAN_DrugBiLSTM_Features_DA | BiLSTM + CNN | drug | CDAN | 0.6050 | 0.5453 | 0.5070 |
| 13 | DrugBAN_DrugBiLSTM_ProteinBiLSTM_ProteinFeatures_DA | BiLSTM + BiLSTM | protein | CDAN | 0.5922 | 0.5311 | 0.4744 |
| 14 | DrugBAN_DANN | GCN + CNN | | DANN | 0.5899 | 0.5303 | 0.4643 |
| 15 | DrugBAN_MMD | GCN + CNN | | MMD | 0.5527 | 0.4849 | 0.5037 |
| 16 | DrugBAN_DrugBiLSTM_DA | BiLSTM + CNN | | CDAN | 0.5520 | 0.5234 | 0.4649 |

---

## 🎯 

### 1. 

#### ✅ success:

1. **Protein BiLSTM **:
 - original CNNBiLSTM AUROC **+0.38%** (0.9641 vs 0.9603)
 - proteinsequence
 - config

2. ****:
 - drug: GCN > BiLSTM (structureimportant)
 - protein: BiLSTM > CNN (sequenceimportant)
 - group: **GCN (Drug) + BiLSTM (Protein)**

3. **featureanalysis**:
 - featureanalysis
 - baseline modelkeymode
 - 

#### ❌ :

1. **domain adaptationmethod**:
 - DA method (CDAN, DANN, MMD) significant
 - analysis
 - :
 * Cluster split 
 * /
 * DA parameter

2. **MAML row**:
 - Meta-learning method
 - 

### 2. structure

#### Abstract :
- Protein BiLSTM 
- **AUROC 0.9641** 
- analysisdomain adaptationmethod

#### Method :
1. **model**: GCN + BiLSTM group
2. **feature**: physicochemicalfeatureextractuse
3. **domain adaptation**: CDAN, DANN, MMD 

#### Results :
1. **comparison** ( + )
2. **featureanalysis**
3. **DA methodanalysis** (important)
4. **** (ablation study)

#### Discussion :
- **success**: BiLSTM 
- ****: DA method
- ****:
 * 
  * Meta-learning method
 * multi-task

### 3. visualization

#### 1: comparison ()
- X : group
- Y : AUROC
- display: Baseline 

#### 2: domain adaptationmethod (/)
- Baseline vs. CDAN vs. DANN vs. MMD
- 
- type

#### 3: featureanalysis (heatmap)
- row: 
- column: featuretype
- color: AUROC value

#### 4: ROC 
- 3-5 model ROC 
- difference

---

## 📝 

### Results :

```markdown
### Encoder Architecture Comparison

We systematically evaluated different encoder architectures for both drug and
protein representations. As shown in Table X, the combination of Graph Convolutional
Network (GCN) for drugs and Bidirectional LSTM (BiLSTM) for proteins achieved the
best performance with AUROC of 0.9641, outperforming the original GCN-CNN baseline
(AUROC: 0.9603) by 0.38%.

This improvement can be attributed to BiLSTM's ability to capture long-range
dependencies in protein sequences, which are crucial for understanding binding sites
and functional domains. In contrast, CNNs with fixed kernel sizes may miss such
long-range interactions.

### Domain Adaptation Analysis

Interestingly, all domain adaptation methods (CDAN, DANN, and MMD) showed
significantly decreased performance compared to their respective baselines, with
average AUROC drops of 35.8%, 35.9%, and 37.0% respectively (Table Y). This
unexpected result suggests several possibilities:

1. The cluster-based data split may not represent a true domain shift problem,
   as proteins within different clusters may share similar binding mechanisms.

2. The baseline models are already sufficiently robust to handle the distribution
   differences in our dataset.

3. Domain adaptation methods introduce additional regularization that may be
   detrimental when source and target domains are not substantially different.

Further investigation is needed to determine whether alternative domain definitions
or meta-learning approaches could provide better generalization.
```

---

## 🔬 statistics

### rowstatisticstest:

1. ** t **:
 - BiLSTM vs. CNN (protein)
 - Baseline vs. DA method

2. **ANOVA + post-hoc**:
 - groupcomparison

3. ** (Effect Size)**:
 - Cohen's d computeactual

4. ****:
   - report 95% CI for AUROC/AUPRC

---

## 📚 

:

1. **BiLSTM for sequences**: Graves & Schmidhuber (2005), Hochreiter & Schmidhuber (1997)
2. **GCN for molecules**: Kipf & Welling (2017), Duvenaud et al. (2015)
3. **Domain Adaptation**:
   - CDAN: Long et al. (2018)
   - DANN: Ganin et al. (2016)
   - MMD: Tzeng et al. (2014)
4. **DTI prediction**: Öztürk et al. (2018 - DeepDTA), Nguyen et al. (2021 - GraphDTA)

---

## 💡 

1. ****:
 - drugproteincluster
 - 

2. **Meta-Learning**:
 - MAML (current)
   - Prototypical Networks
   - Meta-SGD

3. **multi-task**:
 - prediction binding affinity binding sites
 - 

4. **attentionanalysis**:
 - visualizationmodelimportantregion
 - align

---

**datafile**:
- result: `all_experiments_results.csv`
- originaloutput: `result/*/test_markdowntable.txt`

**update**: 2025-12-18
