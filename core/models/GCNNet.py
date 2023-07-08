import torch
import torch.nn as nn
from torch_geometric.nn import GraphNorm ,GCNConv, global_mean_pool as gep, global_max_pool as gmp
from core.edge_gcn import GCNEdgeConv

# Global CONCATENATION
# Double GCN based model
class GCNEdgeNet(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=66, num_features_xt=1280, latent_dim=128, output_dim=128, dropout=0.2, edge_input_dim=None):
        super(GCNEdgeNet, self).__init__()
        self.n_output = n_output
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # SMILES graph branch
        self.dconv1 = GCNEdgeConv(num_features_xd, num_features_xd, edge_input_dim=edge_input_dim, add_self_loops=False)
        self.dconv2 = GCNEdgeConv(num_features_xd, num_features_xd*2, edge_input_dim=edge_input_dim, add_self_loops=False)
        self.dconv3 = GCNEdgeConv(num_features_xd*2, num_features_xd * 4, edge_input_dim=edge_input_dim, add_self_loops=False)
        self.fc_gd1 = torch.nn.Linear(num_features_xd*4, 1024)
        self.fc_gd2 = torch.nn.Linear(1024, output_dim)

        # protein sequence branch (1d conv)
        self.tconv1 = GCNConv(num_features_xt, latent_dim, add_self_loops=False) #1024
        self.tconv2 = GCNConv(latent_dim, latent_dim*2, add_self_loops=False) #512
        self.tconv3 = GCNConv(latent_dim*2, latent_dim*4, add_self_loops=False) #256

        # Global feature branch
        self.glob_linear1 = torch.nn.Linear(num_features_xt, 1024)
        self.glob_linear2 = torch.nn.Linear(1024, 512)
        self.glob_linear3 = torch.nn.Linear(512, output_dim)

        # Protein refinement phase
        self.fc_xt1 = nn.Linear(latent_dim*4, 1024)
        self.fc_xt2 = nn.Linear(1024, output_dim)

        # combined layers
        self.fc1 = nn.Linear(3*output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)

        # Norm layers
        self.DGnorm1 = GraphNorm(num_features_xd)
        self.DGnorm2 = GraphNorm(num_features_xd*2)
        self.DGnorm3 = GraphNorm(num_features_xd*4)
        self.TGnorm1 = GraphNorm(latent_dim)
        self.TGnorm2 = GraphNorm(latent_dim*2)
        self.TGnorm3 = GraphNorm(latent_dim*4)
        self.batchnorm1 = nn.BatchNorm1d(1024)
        self.batchnorm2 = nn.BatchNorm1d(1024)
        self.batchnorm3 = nn.BatchNorm1d(1024)
        self.batchnorm4 = nn.BatchNorm1d(512)
        self.batchnorm5 = nn.BatchNorm1d(1024)
        self.batchnorm6 = nn.BatchNorm1d(512)
        self.batchnorm7 = nn.BatchNorm1d(512)

    def forward(self, data_mol, data_prot):
        # get graph input
        x, edge_index, batch, edge_attr = data_mol.x, data_mol.edge_index, data_mol.batch, data_mol.edge_attr
        # get protein input
        target_x, target_edge_index, target_batch, target_x_global = data_prot.x, data_prot.edge_index, data_prot.batch, data_prot.x_global
    
        x = self.dconv1(x, edge_index, edge_attr)
        #x = self.relu(x)
        x = self.relu(self.DGnorm1(x))

        x = self.dconv2(x, edge_index, edge_attr) 
        #x = self.relu(x)
        x = self.relu(self.DGnorm2(x))

        x = self.dconv3(x, edge_index, edge_attr)
        #x = self.relu(x)
        x = self.relu(self.DGnorm3(x))
        x = gep(x, batch) # global mean pooling

        # flatten
        x = self.batchnorm1(self.fc_gd1(x))
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc_gd2(x)
        #x = self.dropout(x)

        # target protein
        xt = self.tconv1(target_x, target_edge_index)
        #xt = self.relu(xt)
        xt = self.relu(self.TGnorm1(xt))

        xt = self.tconv2(xt, target_edge_index)
        #xt = self.relu(xt)
        xt = self.relu(self.TGnorm2(xt))

        xt = self.tconv3(xt, target_edge_index)
        #xt = self.relu(xt)
        xt = self.relu(self.TGnorm3(xt))
        xt = gep(xt, target_batch) # global mean pooling
  
        # flatten
        xt = self.batchnorm2(self.fc_xt1(xt))
        xt = self.relu(xt)
        xt = self.dropout(xt)
        xt = self.fc_xt2(xt)
        #xt = self.dropout(xt)

        # Global feature
        xg = self.glob_linear1(target_x_global)
        xg = self.batchnorm5(xg)
        xg = self.relu(xg)
        xg = self.glob_linear2(xg)
        xg = self.batchnorm6(xg)
        xg = self.relu(xg)
        xg = self.glob_linear3(xg)

        # concat
        xc = torch.cat((x, xt, xg), 1)
        # add some dense layers
        xc = self.batchnorm3(self.fc1(xc))
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.batchnorm4(self.fc2(xc))
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out

class GCNNet(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=69, num_features_xt=33, latent_dim=128, output_dim=128, dropout=0.2):
        super(GCNNet, self).__init__()

        # SMILES graph branch
        self.n_output = n_output
        self.dconv1 = GCNConv(num_features_xd, num_features_xd, add_self_loops=False)
        self.dconv2 = GCNConv(num_features_xd, num_features_xd*2, add_self_loops=False)
        self.dconv3 = GCNConv(num_features_xd*2, num_features_xd*4, add_self_loops=False)
        self.fc_gd1 = torch.nn.Linear(num_features_xd*4, 1024)
        self.fc_gd2 = torch.nn.Linear(1024, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # protein sequence branch (1d conv)
        self.tconv1 = GCNConv(num_features_xt, latent_dim, add_self_loops=False) #1024
        self.tconv2 = GCNConv(latent_dim, latent_dim*2, add_self_loops=False) #512
        self.tconv3 = GCNConv(latent_dim*2, latent_dim*4, add_self_loops=False) #256
        self.fc_xt1 = nn.Linear(latent_dim*4, 1024)
        self.fc_xt2 = nn.Linear(1024, output_dim)

        # combined layers
        self.fc1 = nn.Linear(2*output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)

        # Norm layers
        self.DGnorm1 = GraphNorm(num_features_xd)
        self.DGnorm2 = GraphNorm(num_features_xd*2)
        self.DGnorm3 = GraphNorm(num_features_xd*4)
        self.TGnorm1 = GraphNorm(latent_dim)
        self.TGnorm2 = GraphNorm(latent_dim*2)
        self.TGnorm3 = GraphNorm(latent_dim*4)
        self.batchnorm1 = nn.BatchNorm1d(1024)
        self.batchnorm2 = nn.BatchNorm1d(1024)
        self.batchnorm3 = nn.BatchNorm1d(1024)
        self.batchnorm4 = nn.BatchNorm1d(512)

    def forward(self, data_mol, data_prot):
        # get graph input
        x, edge_index, batch, edge_attr = data_mol.x, data_mol.edge_index, data_mol.batch, data_mol.edge_attr
        # get protein input
        target_x, target_edge_index, target_batch = data_prot.x, data_prot.edge_index, data_prot.batch

        x = self.dconv1(x, edge_index)
        x = self.relu(self.DGnorm1(x))

        x = self.dconv2(x, edge_index)
        x = self.relu(self.DGnorm2(x))

        x = self.dconv3(x, edge_index)
        x = self.relu(self.DGnorm3(x))
        x = gep(x, batch) # global mean pooling

        # flatten
        x = self.batchnorm1(self.fc_gd1(x))
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc_gd2(x)
        x = self.dropout(x)

        # target protein
        xt = self.tconv1(target_x, target_edge_index)
        xt = self.relu(self.TGnorm1(xt))

        xt = self.tconv2(xt, target_edge_index)
        xt = self.relu(self.TGnorm2(xt))

        xt = self.tconv3(xt, target_edge_index)
        xt = self.relu(self.TGnorm3(xt))
        xt = gep(xt, target_batch) # global mean pooling

        # flatten
        xt = self.batchnorm2(self.fc_xt1(xt))
        xt = self.relu(xt)
        xt = self.dropout(xt)
        xt = self.fc_xt2(xt)
        xt = self.dropout(xt)

        # concat
        xc = torch.cat((x, xt), 1)
        # add some dense layers
        xc = self.batchnorm3(self.fc1(xc))
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.batchnorm4(self.fc2(xc))
        xc = self.relu(xc)
        #xc = self.dropout(xc)
        out = self.out(xc)
        return out