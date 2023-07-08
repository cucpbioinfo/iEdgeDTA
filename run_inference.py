from torch_geometric.loader import DataLoader
from core.data_processing import prepare_dataset
from core.utils import *
from core.emetrics import *
from core.models.GCNNet import GCNEdgeNet
import torch
import argparse
import pathlib
import pandas as pd

def create_parser():
    parser = argparse.ArgumentParser(
        description="iEdgeDTA inference pipeline"
    )

    parser.add_argument(
        "-m",
        "--model-path",
        type=str,
        help="Path to pytorch model checkpoint file (*.model), (default=%(default)s)",
        default="weights/model_edgeGCNNet_withEdge_FOLD0.model"
    )
    parser.add_argument(
        "-s",
        "--smile-path",
        type=pathlib.Path,
        help="SMILEs file on which to predict the affinity (*.fasta)",
        required=True
    )
    parser.add_argument(
        "-f",
        "--fasta-path",
        type=pathlib.Path,
        help="FASTA file on which to predict the affinity (*.fasta)",
        required=True
    )
    parser.add_argument(
        "-p",
        "--feature-dir",
        type=pathlib.Path,
        help="Path to directory that contain the protein embedding feature, (default=%(default)s)",
        default="dataset/protein_features/"
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=pathlib.Path,
        help="Output directory for prediction and other intermediate result, (default=%(default)s)",
        default="inference-result"
    )
    return parser

def GET_MODEL():
    model = GCNEdgeNet(num_features_xd=66, num_features_xt=1280, dropout=0, edge_input_dim=18)
    model_st = GCNEdgeNet.__name__
    return model, model_st

def LOAD_DATA(data=None, batch_size=1):
    data_loader = None
    if data != None:
        data_loader = DataLoader(data, batch_size=batch_size, shuffle=False, collate_fn=collate, pin_memory=True)
    return data_loader

def WRITE_CSV(get_path, save_path, pred):
    df = pd.read_csv(get_path)
    df["affinity_score"] = pred
    df.to_csv(f'{save_path}/prediction.csv', index=False)
    return f'{save_path}/prediction.csv'

def RUN_INFERENCE(args, inference_data, device):

    model, model_st = GET_MODEL()
    model = model.to(device)
    model.load_state_dict(torch.load(args.model_path))

    data_loader = LOAD_DATA(data=inference_data, batch_size=1)
    
    print("Inferencing . . . .\n")
    
    aff_pred = inference(model, device, data_loader)
    csv_path = os.path.join(args.output_dir, "processed", "dta_pair.csv")
    save_path = args.output_dir
    output = WRITE_CSV(csv_path, save_path, aff_pred)

    print("\n. . . Inferencing Complete . . .\n")
    print(f"Prediction result : {output}")

if __name__=="__main__":
    parser = create_parser()
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Inference on: ', device, '\n')

    inference_data = prepare_dataset(
        smile_path=args.smile_path,
        fasta_path=args.fasta_path,
        feature_path=args.feature_dir,
        output_dir=args.output_dir,
        windows=3
    )
    
    RUN_INFERENCE(
        args=args,
        inference_data=inference_data,
        device=device
    )

