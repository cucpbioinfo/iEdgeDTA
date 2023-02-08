#!/bin/bash
python emb_esm.py pretrained/esm1v_t33_650M_UR90S_1.pt dataset/original/sample_proteins.fasta dataset/esm_emb/
python run_inference.py
