# iEdgeDTA
Integrated edge information and 1D graph convolutional neural networks for binding affinity prediction

### Update Apr 2023

Currently source code is in very hard coding manners. we will fix and refactor this later for the ease of used.

For training dataset. please refer to [DeepDTA](https://github.com/hkmztrk/DeepDTA/blob/master/data/README.md)

TODO

There is a huge conflict between training code and inference code that use the same util function that made the current version support only inference.

- [ ] Make it trainable
- [ ] Refactoring all code

## Inference pipeline

Create virtual environment for python=3.8.0

`conda create -n "env-name" python=3.8.0`

install require package simply run

`pip3 install -r requirement.txt`

Please provide protein sequence in **fasta format (.fasta)** and SMILE sequence in **dictionary format (.txt)**

You can find an example in `dataset/original/sample_*`

## Runing inference

***Option 1: bash script (full pipeline)***

Simply run `bash run.sh`

***Option 2: run separate module***

```bash
$python emb_esm.py pretrained/esm1v_t33_650M_UR90S_1.pt dataset/original/proteins.fasta dataset/esm_emb/
```
Then run
```bash
$python run_inference.py
```

Output will be saved in `dataset/prediction.csv` as csv file.