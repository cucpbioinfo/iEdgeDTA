#!/bin/bash
# change pretrained/esm1v_t33_650M_UR90S_1.pt to esm1v_t33_650M_UR90S_1 if you want it download automatically.
python emb_esm.py pretrained/esm1v_t33_650M_UR90S_1.pt dataset/original/proteins.fasta dataset/esm_emb/
python run_inference.py
