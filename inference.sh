#!/bin/bash
# --model-location will download automatically, if you want to use local model (.pt)
# that downloaded from esm repository please specify path instead (path/to/model.pt)
python run_esm.py \
    --model-location esm1v_t33_650M_UR90S_1 \
    --fasta-file dataset/sample_inference/original/proteins.fasta \
    --output-dir dataset/protein_feature/

python run_inference.py \
   --model-path weights/model_final_davis.model \
   --smile-path dataset/sample_inference/original/ligands.txt \
   --fasta-path dataset/sample_inference/original/proteins.fasta \
   --feature-dir dataset/protein_feature/ \
   --output-dir inference-result/ 
