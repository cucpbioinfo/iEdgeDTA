import os
import re
import numpy as np
import pandas as pd
from collections import OrderedDict
import json, pickle
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
from tqdm import tqdm
from core.CreateDataset_infer import CreateDatasetInfer
from core.CreateDataset_train import CreateDatasetTrain    
import networkx as nx   
import torch
import torchdrug as td
from torchdrug.data import Molecule
from typing import Dict
from Bio import SeqIO
import datetime
from typing import List, Dict, Tuple, Union

#################################### LIGAND PROCESSING ################################

def Node_Feature(mol: Molecule):
    """
        :param Molecule mol: predicted molecule features from SMILEs
        :rtype: np.ndarray node feature from molecule object
    """
    return mol.node_feature

def Edge_Feature(mol: Molecule):
    """
        :param Molecule mol: predicted molecule features from SMILEs
        :rtype: np.ndarray edge feature from molecule object
    """
    e_feat = mol.edge_feature
    return e_feat

def Adj_list(mol: Molecule):
    """
        :param Molecule mol: predicted molecule features from SMILEs
        :rtype: np.ndarray adjacency list from molecule object
    """
    return mol.edge_list[:, :2]

def add_self_loop(node_list: np.ndarray, edge_list: np.ndarray, edge_attr: np.ndarray):
    """
        This function will append the self loop in adjacency list, and edge attribute list
    """
    self_edge_index = []
    self_edge_attr = []
    edge_attr_dim = edge_attr.shape[1]
    self_loop = np.matrix(np.eye(node_list.shape[0]))
    index_row, index_col = np.where(self_loop >= 0.5)
    for i, j in zip(index_row, index_col):
        self_edge_index.append([i, j])
        if edge_attr is not None:
            self_edge_attr.extend(np.ones([1, edge_attr_dim], int).tolist())
    edge_list = np.append(edge_list, self_edge_index, axis=0)   
    edge_attr = np.append(edge_attr, self_edge_attr, axis=0)
    return edge_list, edge_attr

def combine_feature(smile: str):
    """
        This function will combine all neccesity feature in to an array
        :param str smile: compound string in SMILEs format
        :rtype List: a list of feature [the last dim just a place holder for edge type for experiment]
    """
    x = np.array(Node_Feature(smile))
    y = np.array(Edge_Feature(smile))
    z, y = add_self_loop(x, Adj_list(smile), y)
    #t = np.array(Edge_type(smile)) For RGCN
    return [x, y, z, [0]] 

def make_ligand_feature(path: str, split: str='train', dataset: str='davis', debug: bool=False):
    """
        This function will create a dictionary of drug feature and save the data to csv
        with the format "{dataset-name}_{split}.csv"
    """
    filename = f"{dataset}_{split}.csv"
    path_to_csv = os.path.join(path, filename)
    df = pd.read_csv(path_to_csv)

    print("###########################")
    print(" MAKE FEATURES ON " , split, " Dataset")
    print("###########################\n")
    smile_list = set(np.array(df.compound_id))
    print("Number of unique SMILEs : ", len(smile_list))
    smile_graph = {}

    if debug==True:
        for smile in smile_list:
            mol = td.data.Molecule.from_smiles(smile)
            feature = combine_feature(mol)
            smile_graph[smile] = feature  
        return smile_graph

    for smile_id in smile_list:
        try:
            smile = df[df["compound_id"] == smile_id]["compound_smiles"].values[0]
            mol = td.data.Molecule.from_smiles(smile)
            feature = combine_feature(mol)
            smile_graph[smile_id] = feature
        except:
            # Raise from "Invalid: SMILE"
            # This should be also removed from dataframe
            df = df.drop(df[df['compound_smiles']==smile].index)
            continue
    # Save for utilizing as tracking in smile_graph in future process.
    save_path = os.path.join(path, "processed", filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print("Saving Modified Dataframe to file . . .")
    print("Saving location : ", save_path, "\n\n")
    return smile_graph

#################################### PROTEIN PROCESSING ##################################

def create_pseudo_graph(len_seq: int, windows: int=3):
    """
        Create a pseudo graph from sequence length
        :rtype : List[List] adjacency list
    """
    # This function already add self loop.
    assert windows % 2 == 1, "windows should be odds number."
    assert windows <= len_seq, "windows must less than sequence length."
    center = (windows//2) # Indice start from 0, +1 if indice start from 1
    adj = []
    for seq in range(len_seq):
        mid = seq+center
        if mid == (len_seq-center):
            break
        adj.extend([mid, target+seq] for target in range(windows))
    # Adding before windows edges.
    for i in range(center):
        adj.extend([i, target] for target in range(center+i+1))
    # Adding After windows edges.   
    for i in range(center):
        index = len_seq-i-1
        adj.extend([index, len_seq-1-target] for target in range(center+i+1))
    return adj

def prot_to_graph(path: str, path_to_feature: str, split: str='full', dataset: str='davis', windows: int=3):
    """
        Function time_complexity O(n) ; n = number of protein in list
        prot_key(List) : List of Protein
        return dictionary of {protein key : its features}
    """
    filename = f"{dataset}_{split}.csv"
    path_to_csv = os.path.join(path, filename)
    df = pd.read_csv(path_to_csv)

    print("###########################")
    print(" MAKE PROTEIN GRAPH ON " , split, " Dataset")
    print("###########################\n")
    prot_key = set(np.array(df.target_id))
    print("Number of unique Protein : ", len(prot_key))
    prot_graph = {}
    
    for key in prot_key:
        emb = torch.load(os.path.join(path_to_feature, f"{key}.pt"))
        target_feature = emb["representations"].squeeze()
        target_global_feature = emb["sequence_repr"]
        target_size = len(target_feature)
        target_edge_index = create_pseudo_graph(target_size, windows)

        prot_graph[key] = [target_size, target_feature, target_edge_index, target_global_feature]
    print("Protein graph construction -> Complete  \( ﾟヮﾟ)/\n\n")
    return prot_graph

################################### PREPARE DATASET ##################################

def prepare_dataset_withFolds(dataset: str, path: str, feature_path:str, fold: int=0, windows: int=3):
    assert(type(dataset) == str), "InputFailed: datasets should be string (davis, kiba, etc.)."

    print('Convert data for ', dataset)
    fpath = os.path.join(path, dataset)

    train_folds = json.load(open(os.path.join(fpath, "original", "folds/train_fold_setting1.txt")))
    train_folds = [folds for folds in train_folds] # (folds, N/folds)

    train_fold = []
    valid_fold = train_folds[fold]
    for i in range(len(train_folds)):
        if i != fold:
            train_fold += train_folds[i]

    test_fold = json.load(open(os.path.join(fpath, "original", "folds/test_fold_setting1.txt")))
    ligands = json.load(open(os.path.join(fpath, "original", "ligands_can.txt")), object_pairs_hook=OrderedDict)
    proteins = json.load(open(os.path.join(fpath, "original", "proteins.txt")), object_pairs_hook=OrderedDict)
    affinity = pickle.load(open(os.path.join(fpath, "original", "Y"),"rb"), encoding='latin1')

    drugs = []
    prots = []
    prot_key = []
    for d in ligands.keys():
        lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]),isomericSmiles=True)
        drugs.append({"id":d , "smile":lg})
    for t in proteins.keys():
        prot_key.append(t)
        prots.append(proteins[t])
    if dataset == 'davis':
        affinity = [-np.log10(y/1e9) for y in affinity]

    affinity = np.asarray(affinity)
    opts = ['train','test','validation','full']
    for opt in opts:
        rows, cols = np.where(np.isnan(affinity)==False)  
        if opt=='train':
            rows, cols = rows[train_fold], cols[train_fold]
        elif opt=='test':
            rows, cols = rows[test_fold], cols[test_fold]
        elif opt=='validation':
            rows, cols = rows[valid_fold], cols[valid_fold]
        with open(os.path.join(fpath, f"{dataset}_{opt}.csv"), "w") as f:
            f.write('compound_id,compound_smiles,target_id,target_sequence,affinity\n')
            for pair_ind in range(len(rows)):
                ls = []
                ls += [ drugs[rows[pair_ind]].get("id") ]
                ls += [ drugs[rows[pair_ind]].get("smile") ]
                ls += [ prot_key[cols[pair_ind]] ]
                ls += [ prots[cols[pair_ind]]  ]
                ls += [ affinity[rows[pair_ind],cols[pair_ind]]  ]
                f.write(','.join(map(str,ls)) + '\n')        

    set_drugs = [drug_id for drug_id, _ in drugs]
    print('\ndataset:', dataset)
    print('train_fold:', len(train_fold))
    print('valid_fold:', len(valid_fold), " on fold: ", fold)
    print('test_fold:', len(test_fold))
    print('len(set(drugs)),len(set(prots)):', len(set(set_drugs)), ", ", len(set(prots)))
    print("#####################################################################################\n\n")
    print("Data folds prepared, Graph processing in progress . . .\n\n")

    smile_graph_train = make_ligand_feature(path=fpath, split='train', dataset=dataset, debug=False) #This process will skip missing SMILES
    smile_graph_valid = make_ligand_feature(path=fpath, split='validation', dataset=dataset, debug=False) #This process will skip missing SMILES
    smile_graph_test = make_ligand_feature(path=fpath, split='test', dataset=dataset, debug=False) #This process will skip missing SMILES

    prot_graph_train = prot_to_graph(path=fpath, path_to_feature=feature_path, split='train', dataset=dataset, windows=windows)
    prot_graph_valid = prot_to_graph(path=fpath, path_to_feature=feature_path, split='validation', dataset=dataset, windows=windows)
    prot_graph_test = prot_to_graph(path=fpath, path_to_feature=feature_path, split='test', dataset=dataset, windows=windows)
    
    print("#####################################################################################\n\n")
    df = pd.read_csv(os.path.join(fpath, "processed", f"{dataset}_train.csv"))
    train_drugs = list(df.compound_id)
    train_prots = list(df.target_id)
    train_aff = list(df.affinity)
    train_drugs, train_prots, train_Y = np.asarray(train_drugs), np.asarray(train_prots), np.asarray(train_aff)

    df = pd.read_csv(os.path.join(fpath, "processed", f"{dataset}_validation.csv"))
    valid_drugs = list(df.compound_id)
    valid_prots = list(df.target_id)
    valid_aff = list(df.affinity)
    valid_drugs, valid_prots, valid_Y = np.asarray(valid_drugs), np.asarray(valid_prots), np.asarray(valid_aff)

    df = pd.read_csv(os.path.join(fpath, "processed", f"{dataset}_test.csv"))
    test_drugs = list(df.compound_id)
    test_prots = list(df.target_id)
    test_aff = list(df.affinity)
    test_drugs, test_prots, test_Y = np.asarray(test_drugs), np.asarray(test_prots), np.asarray(test_aff)

    root_path = os.path.join(fpath, "processed")
    print('preparing ', dataset + '_train pytorch format!')
    train_data = CreateDatasetTrain(root=root_path,
                            dataset=dataset+'_train',
                            drugList=train_drugs,
                            protkey=train_prots,
                            y=train_Y,
                            smile_graph=smile_graph_train,
                            protein_graph=prot_graph_train
                            )

    print('preparing ', dataset + '_validation pytorch format!')
    valid_data = CreateDatasetTrain(root=root_path,
                            dataset=dataset+'_validation',
                            drugList=valid_drugs,
                            protkey=valid_prots,
                            y=valid_Y,
                            smile_graph=smile_graph_valid,
                            protein_graph=prot_graph_valid
                            )

    print('preparing ', dataset + '_test in pytorch format!')
    test_data = CreateDatasetTrain(root=root_path,
                            dataset=dataset+'_test',
                            drugList=test_drugs,
                            protkey=test_prots,
                            y=test_Y,
                            smile_graph=smile_graph_test,
                            protein_graph=prot_graph_test
                            )
    print('\nPytorch dataset have been created  \( ﾟヮﾟ)/ HooRay!!')   

    return train_data, valid_data, test_data


#################################### INFERENCE UTIL ####################################

def fasta2dict(filepath: str):
    """
        Read fasta format
    """
    seq_dict = {rec.id : rec.seq for rec in SeqIO.parse(filepath, "fasta")}
    return seq_dict

def drug_target_mapping(drug_info: Dict[str, str], target_info: Dict[str, str]):
    """
        Map all drug and target protein to all-pair dataset
    """   
    drug_ids = []
    drug_sequences = []
    target_ids = []
    target_sequences = []
    for drug_id in drug_info.keys():
        for target_id in target_info.keys():
            drug_ids.append(drug_id)
            drug_sequences.append(drug_info.get(drug_id))
            target_ids.append(target_id)
            target_sequences.append(target_info.get(target_id))
    return drug_ids, drug_sequences, target_ids, target_sequences

def prepare_dataset(smile_path: str, fasta_path: str, feature_path: str, output_dir: str, windows=3):
    
    # GET DATETIME
    timezone = datetime.datetime.strptime("+0700", "%z").tzinfo
    current_date = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).astimezone(timezone)
    date = str(current_date).split(" ")[0]

    print('Preparing data for inference . .')

    ligands_path = smile_path
    proteins_path = fasta_path

    drugs = json.load(open(ligands_path), object_pairs_hook=OrderedDict)
    prots = fasta2dict(proteins_path)

    drug_ids, drug_sequences, target_ids, target_sequences = drug_target_mapping(drugs, prots)
    # TODO : Create csv from list
    csv_dict = {"compound_id" : drug_ids, "compound_smiles" : drug_sequences, "target_id" : target_ids, "target_sequence" : target_sequences}
    df = pd.DataFrame(csv_dict)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}", exist_ok=True)
    df.to_csv(f'{output_dir}/dta_pair.csv')
           
    print('\nInference on:', date)
    print('Unique drugs, Unique proteins:', len(set(drugs)), ", ", len(set(prots)))
    print("#####################################################################################\n\n")
    print("Data prepared, Graph processing in progress . . .\n\n")

    path = output_dir
    smile_graph = make_ligand_feature(path = path, split="pair", dataset="dta", debug=False) #This process will skip missing SMILES
    prot_graph = prot_to_graph(path = path, path_to_feature=feature_path, split="pair", dataset="dta", windows=windows)
    
    print("#####################################################################################\n\n")

    df = pd.read_csv(os.path.join(output_dir, "processed", "dta_pair.csv"))
    train_drugs = list(df.compound_id)
    train_prots = list(df.target_id)
    inference_drugs, inference_prots = np.asarray(train_drugs), np.asarray(train_prots)

    root_path = os.path.join(output_dir, "processed")
    inference_data = CreateDatasetInfer(root=root_path,
                            dataset='data_inference',
                            drugList=inference_drugs,
                            protkey=inference_prots,
                            smile_graph=smile_graph,
                            protein_graph=prot_graph
                            )

    print('\nPytorch dataset have been created  \( ﾟヮﾟ)/ HooRay!!')   

    return inference_data