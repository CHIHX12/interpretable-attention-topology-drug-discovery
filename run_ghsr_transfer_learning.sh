#!/bin/bash

################################################################################
# GHSR transfer learningtraining
# Transfer Learning from BindingDB to GHSR
#
# 
# 1. BindingDB training DrugBAN_BiLSTM model (epoch 94)
# 2. GHSR datarowfine-tuning
# 3. usesplit train/val/test data
#
# 
# - transfer learningtraining
# - data
################################################################################

# color
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "================================================================================"
echo "🧬 GHSR transfer learning - Transfer Learning from BindingDB to GHSR"
echo "================================================================================"
echo ""

# checkfile
echo -e "${YELLOW}📋 checkfile...${NC}"

if [ ! -f "configs/DrugBAN_BiLSTM_GHSR_TransferLearning.yaml" ]; then
 echo -e "${RED}❌ configfile: configs/DrugBAN_BiLSTM_GHSR_TransferLearning.yaml${NC}"
    exit 1
fi

if [ ! -f "result/DrugBAN_BiLSTM/best_model_epoch_94.pth" ]; then
 echo -e "${RED}❌ trainingmodel: result/DrugBAN_BiLSTM/best_model_epoch_94.pth${NC}"
 echo -e "${YELLOW} BindingDB trainingmodel${NC}"
    exit 1
fi

if [ ! -d "datasets/GPCR_resarch/random" ]; then
 echo -e "${RED}❌ GHSR splitdata: datasets/GPCR_resarch/random/${NC}"
 echo -e "${YELLOW} row: python3 split_ghsr_data.py --stratify${NC}"
    exit 1
fi

echo -e "${GREEN}✓ file${NC}"
echo ""

# displaydata
echo -e "${YELLOW}📊 GHSR data:${NC}"
echo "   Train:  $(tail -n +2 datasets/GPCR_resarch/random/train.csv | wc -l) samples"
echo "   Val:    $(tail -n +2 datasets/GPCR_resarch/random/val.csv | wc -l) samples"
echo "   Test:   $(tail -n +2 datasets/GPCR_resarch/random/test.csv | wc -l) samples"
echo ""

# displayconfig
echo -e "${YELLOW}⚙️  trainingconfig:${NC}"
echo "   configfile: configs/DrugBAN_BiLSTM_GHSR_TransferLearning.yaml"
echo " trainingmodel: result/DrugBAN_BiLSTM/best_model_epoch_94.pth"
echo "   outputdirectory: result/DrugBAN_BiLSTM_GHSR_TransferLearning/"
echo " learning rate: 1e-5 (fine-tuninglearning rate)"
echo "   Batch Size: 32"
echo "   Max Epoch: 50"
echo ""

# confirm
read -p "starttransfer learningtraining？(y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
 echo "training"
    exit 0
fi

echo ""
echo "================================================================================"
echo "🚀 starttraining..."
echo "================================================================================"
echo ""

# rowtraining
python3 main.py \
    --cfg configs/DrugBAN_BiLSTM_GHSR_TransferLearning.yaml \
    --data GPCR_resarch \
    --split random

# checktrainingresult
if [ $? -eq 0 ]; then
    echo ""
    echo "================================================================================"
    echo -e "${GREEN}✅ trainingdone！${NC}"
    echo "================================================================================"
    echo ""

    # displayresult
    if [ -d "result/DrugBAN_BiLSTM_GHSR_TransferLearning" ]; then
        echo -e "${YELLOW}📁 outputdirectory:${NC}"
        echo "   result/DrugBAN_BiLSTM_GHSR_TransferLearning/"
        echo ""

 # columngeneratemodel
        if ls result/DrugBAN_BiLSTM_GHSR_TransferLearning/*.pth 1> /dev/null 2>&1; then
            echo -e "${YELLOW}🎯 trainingmodel:${NC}"
            ls -lh result/DrugBAN_BiLSTM_GHSR_TransferLearning/*.pth
            echo ""
        fi

        # displayresultfile
        if [ -f "result/DrugBAN_BiLSTM_GHSR_TransferLearning/result.txt" ]; then
            echo -e "${YELLOW}📊 trainingresult:${NC}"
            cat result/DrugBAN_BiLSTM_GHSR_TransferLearning/result.txt
            echo ""
        fi
    fi

    echo -e "${YELLOW}📝 next steps:${NC}"
    echo ""
 echo "1. usetrainingmodelrowpredictionattentionanalysis"
    echo "   python3 batch_predict_ghsr.py \\"
    echo "       --config configs/DrugBAN_BiLSTM_GHSR_TransferLearning.yaml \\"
    echo "       --model_path result/DrugBAN_BiLSTM_GHSR_TransferLearning/best_model_epoch_XX.pth \\"
    echo "       --output_dir datasets/GPCR_resarch/attention_results_TransferLearning"
    echo ""
 echo "2. comparisontransfer learningtrainingdifference"
    echo ""

else
    echo ""
    echo "================================================================================"
 echo -e "${RED}❌ training${NC}"
    echo "================================================================================"
    echo ""
    exit 1
fi
