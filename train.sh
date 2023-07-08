#!/bin/bash
# --model-location will download automatically, if you want to use local model (.pt)
# that downloaded from esm repository please specify path instead (path/to/model.pt)
DATASET=davis
python run_esm.py \
    --model-location esm1v_t33_650M_UR90S_1 \
    --fasta-file dataset/$DATASET/original/${DATASET}FASTA.fasta \
    --output-dir dataset/protein_feature/

python train.py --config default_config.json
