from torch_geometric.loader import DataLoader
from core.data_process_1DGCN import prepare_dataset
from core.utils import *
from core.emetrics import *
from core.models.GCNNet import GCNEdgeNet
import torch
import pandas as pd

def GET_MODEL():
    #model = SelfAttentionNet(num_features_xd=78, num_features_xt=1280, dropout=0.1)
    #model_st = SelfAttentionNet.__name__
    model = GCNEdgeNet(num_features_xd=66, num_features_xt=1280, dropout=0, edge_input_dim=22)
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

def RUN_INFERENCE(inference_data, device):

    model, model_st = GET_MODEL()
    model = model.to(device)
    model.load_state_dict(torch.load("weights/model_edgeGCNNet_withEdge_FOLD1.model"))

    data_loader = LOAD_DATA(data=inference_data, batch_size=1)
    
    print("Inferencing . . . .\n")
    
    aff_pred = inference(model, device, data_loader)
    csv_path = f"dataset/processed/dta_pair.csv"
    save_path = f"dataset"
    output = WRITE_CSV(csv_path, save_path, aff_pred)

    print("\n. . . Inferencing Complete . . .\n")
    print(f"Prediction result : {output}")

if __name__=="__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Inference on: ', device, '\n')

    inference_data = prepare_dataset(path="dataset/", windows=3)
    RUN_INFERENCE(inference_data=inference_data, device=device)

