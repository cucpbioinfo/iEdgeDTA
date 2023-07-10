# iEdgeDTA
Integrated edge information and 1D graph convolutional neural networks for binding affinity prediction

### Update Jul 2023

~~Currently source code is in very hard coding manners. we will fix and refactor this later for the ease of used.~~

For training dataset information. please refer to [DeepDTA](https://github.com/hkmztrk/DeepDTA/blob/master/data/README.md)

TODO

~~There is a huge conflict between training code and inference code that use the same util function that made the current version support only inference.~~

- [x] Make it trainable
- [x] Refactoring all code
- [ ] test/update dependencies version (torch, torch-scatter, torch-geometric, torchdrug)

## Setting up (This setup is tested on Ubuntu Focal/Jammy)

Create virtual environment for python=3.8.0

`conda create -n "${env_name}" python=3.8.0`
`conda activate "${env_name}"`
install require package simply run

`pip3 install -r requirement.txt`

## Training

Both training dataset is in `dataset/` directory

See an example of how to train in `train.sh`

- first we need to extract the protein feature by using `run_esm.py` python script

example

```
$python run_esm.py \
    --model-location esm1v_t33_650M_UR90S_1 \
    --fasta-file dataset/davis/original/davisFASTA.fasta \
    --output-dir dataset/protein_feature/
```

```
--model-location : path to pretrained model OR name of pretrained model (this will store in .cache)
--fasta-file : path to protein fasta file
--output-dir : output directory of protein feature
```

- then we are ready to train by using `train.py` and specify the parameter in `config.json` file (see an example in `default_config.json`)

```
$python train.py --config default_config.json
```

```
# Config definition
"filename" : Name of current task (this name will be the name for model/figure/etc. that will be saved in this running task)
"dataset" : Name of dataset ["davis", "kiba", etc(if provided)],
"dataset_path" : Path to dataset [dataset/],
"FOLD" : FOLD to be run [0,1,2,3,4],
"protein_feature_dir" : Path to protein feature directory (from run_esm.py) [dataset/protein_feature],
"BATCH_SIZE" : Batch size,
"NUM_EPOCHS" : Number of epoch,
"max_lr" : Maximum learning rate [0.001],
"lr" : Start learning rate [0.001],
"windows" : Number of adjacency contact in psuedo contact map [3],
"RESUME_TRAIN" : You can resume training from the last saving state by setting this to true [false]

# Note [example of windows parameter]
windows = 3 means the function will construct the graph by connecting adj_left + current_node + adj_right (3)
windows = 5 means the function will construct the graph by connecting 2_adj_left + adj_left + current_node + adj_right + 2_adj_right (5)
```

## Inference pipeline

Please provide protein sequence in **fasta format (.fasta)** and SMILE sequence in **dictionary format (.txt)**

You can find an example in `dataset/sample_inference/*`

## Runing inference

***Option 1: bash script (full pipeline)***

Simply run `inference.sh` (you can find an example of argument in this script)

***Option 2: run separate module (recommended)***

```bash
$python run_esm.py \
    --model-location esm1v_t33_650M_UR90S_1 \
    --fasta-file dataset/sample_inference/original/proteins.fasta \
    --output-dir dataset/protein_feature/
```
Then run
```bash
$python run_inference.py \
   --model-path weights/model_final_davis.model \
   --smile-path dataset/sample_inference/original/ligands.txt \
   --fasta-path dataset/sample_inference/original/proteins.fasta \
   --feature-dir dataset/protein_feature/ \
   --output-dir inference-result/ 
```

Output will be saved in `{output-dir}/prediction.csv` as csv file.
