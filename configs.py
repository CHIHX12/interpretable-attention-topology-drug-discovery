from yacs.config import CfgNode as CN

_C = CN()

# Drug feature extractor
_C.DRUG = CN()
_C.DRUG.NODE_IN_FEATS = 75
_C.DRUG.PADDING = True
_C.DRUG.HIDDEN_LAYERS = [128, 128, 128]
_C.DRUG.NODE_IN_EMBEDDING = 128
_C.DRUG.MAX_NODES = 290

# BiLSTM options (for SELFIES sequence representation)
_C.DRUG.USE_BILSTM = False  # Set True to use BiLSTM (SELFIES) instead of GCN (Graph)
_C.DRUG.VOCAB_SIZE = 100  # SELFIES vocabulary size (auto-computed from dataset at runtime)
_C.DRUG.LSTM_HIDDEN_DIM = 64  # BiLSTM hidden dimension
_C.DRUG.LSTM_NUM_LAYERS = 2  # Number of LSTM layers
_C.DRUG.LSTM_DROPOUT = 0.2  # Dropout rate
_C.DRUG.MAX_DRUG_LENGTH = 200  # Maximum SELFIES sequence length
_C.DRUG.USE_FEATURES = False  # Set True to use drug physicochemical features (only active when USE_BILSTM=True)
_C.DRUG.FEATURE_DIM = 8  # Drug physicochemical feature dimension (MW, logP, TPSA, HBD, HBA, Ring, RotBonds, Charge)

# Protein feature extractor
_C.PROTEIN = CN()
_C.PROTEIN.NUM_FILTERS = [128, 128, 128]
_C.PROTEIN.KERNEL_SIZE = [3, 6, 9]
_C.PROTEIN.EMBEDDING_DIM = 128
_C.PROTEIN.PADDING = True
# BiLSTM options (set True to use BiLSTM instead of CNN)
_C.PROTEIN.USE_BILSTM = False
_C.PROTEIN.LSTM_HIDDEN_DIM = 256
_C.PROTEIN.LSTM_NUM_LAYERS = 2
_C.PROTEIN.LSTM_DROPOUT = 0.2
_C.PROTEIN.FEATURE_DIM = 4  # Amino acid physicochemical feature dimension (hydrophobicity, volume, charge, polarity)
_C.PROTEIN.MAX_PROTEIN_LENGTH = 1200  # Maximum protein sequence length (official default, covers 89% of BindingDB)

# BCN setting
_C.BCN = CN()
_C.BCN.HEADS = 2

# MLP decoder
_C.DECODER = CN()
_C.DECODER.NAME = "MLP"
_C.DECODER.IN_DIM = 256
_C.DECODER.HIDDEN_DIM = 512
_C.DECODER.OUT_DIM = 128
_C.DECODER.BINARY = 1

# SOLVER
_C.SOLVER = CN()
_C.SOLVER.MAX_EPOCH = 100
_C.SOLVER.BATCH_SIZE = 64
_C.SOLVER.NUM_WORKERS = 0
_C.SOLVER.LR = 5e-5
_C.SOLVER.DA_LR = 1e-3
_C.SOLVER.SEED = 2048
# Transfer Learning
_C.SOLVER.PRETRAINED_MODEL = ""  # Path to pretrained model for transfer learning (empty = train from scratch)

# RESULT
_C.RESULT = CN()
_C.RESULT.OUTPUT_DIR = "./result"
_C.RESULT.SAVE_MODEL = True

# Domain adaptation
_C.DA = CN()
_C.DA.TASK = False
_C.DA.METHOD = "CDAN"
_C.DA.USE = False
_C.DA.INIT_EPOCH = 10
_C.DA.LAMB_DA = 1
_C.DA.RANDOM_LAYER = False
_C.DA.ORIGINAL_RANDOM = False
_C.DA.RANDOM_DIM = None
_C.DA.USE_ENTROPY = True

# MAML Meta-Learning configuration
_C.MAML = CN()
_C.MAML.ENABLE = False              # Enable MAML meta-learning
_C.MAML.INNER_LR = 0.01             # Inner loop learning rate (α)
_C.MAML.INNER_STEPS = 10            # Number of inner adaptation steps (K)
_C.MAML.FIRST_ORDER = False         # Use first-order MAML (saves memory)
_C.MAML.TASK_BATCH_SIZE = 4         # Number of tasks per meta-batch
_C.MAML.SUPPORT_SIZE = 16           # Support set size per task
_C.MAML.QUERY_SIZE = 16             # Query set size per task
_C.MAML.MIN_SAMPLES_PER_PROTEIN = 10  # Minimum samples to be considered a task
_C.MAML.MMD_WEIGHT = 0.1            # Weight for MMD loss (λ)
_C.MAML.MMD_KERNEL_MUL = 2.0        # MMD kernel bandwidth multiplier
_C.MAML.MMD_KERNEL_NUM = 5          # Number of kernels for MMD

# Support Set configuration (BatLiNet-inspired few-shot learning)
_C.SUPPORT_SET = CN()
_C.SUPPORT_SET.USE = False  # Enable/disable support set mechanism
_C.SUPPORT_SET.K = 5  # Number of support samples per query
_C.SUPPORT_SET.ALPHA = 0.5  # Weight for support vs original path (0-1)
_C.SUPPORT_SET.SAMPLING_STRATEGY = "random"  # Support sampling strategy
_C.SUPPORT_SET.AGGREGATION_TRAIN = "mean"  # Training aggregation method
_C.SUPPORT_SET.AGGREGATION_TEST = "median"  # Test aggregation method (more robust)

# Multi-task Learning configuration
_C.MULTITASK = CN()
_C.MULTITASK.ENABLED = False  # Enable multi-task learning (classification + regression)
_C.MULTITASK.ALPHA = 0.5  # Weight for classification loss
_C.MULTITASK.BETA = 0.5   # Weight for regression loss
# Total loss = ALPHA * classification_loss + BETA * regression_loss

# Knowledge Distillation configuration
_C.DISTILLATION = CN()
_C.DISTILLATION.ENABLED = False  # Enable knowledge distillation from expert models
_C.DISTILLATION.EXPERT_LOW_MODEL = ""   # Path to Expert_Low model (Z < 0.4 specialist)
_C.DISTILLATION.EXPERT_HIGH_MODEL = ""  # Path to Expert_High model (Z > 0.8 specialist)
_C.DISTILLATION.TEMPERATURE = 3.0       # Softmax temperature for soft targets
_C.DISTILLATION.ALPHA_DISTILL = 0.5     # Weight: (1-α)*ground_truth + α*expert_prediction

# Comet config, ignore it If not installed.
_C.COMET = CN()
# Please change to your own workspace name on comet.
_C.COMET.WORKSPACE = "pz-white"
_C.COMET.PROJECT_NAME = "DrugBAN"
_C.COMET.USE = False
_C.COMET.TAG = None


def get_cfg_defaults():
    return _C.clone()
