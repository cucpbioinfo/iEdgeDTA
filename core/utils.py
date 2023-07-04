import torch
import os
from core.emetrics import *
from torch_geometric.data import Batch
import time
from tqdm.auto import tqdm

def train(model, device, train_loader, optimizer, criterion, epoch):
    total_loss = 0
    training_ci = 0
    model.train()
    for idx, data in enumerate(train_loader):
        data_mol = data[0].to(device)
        data_prot = data[1].to(device)
        optimizer.zero_grad()
        output = model(data_mol, data_prot)    
        label = data_mol.y.view(-1, 1).float().to(device)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        # Change evaluation metric later.
        training_ci += get_ci(output.cpu().detach().numpy(), label.cpu().detach().numpy())
        
    training_ci /= len(train_loader)
    total_loss /= len(train_loader)

    return total_loss, training_ci
        

# Test
def evaluate(model, device, test_loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    #print('Make prediction for {} samples...'.format(len(test_loader.dataset)))
    with torch.no_grad():
        for data in test_loader:
            data_mol = data[0].to(device)
            data_prot = data[1].to(device)
            pred = model(data_mol, data_prot)
            total_preds = torch.cat((total_preds, pred.cpu()), 0)
            total_labels = torch.cat((total_labels, data_mol.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()

def LR_scheduler_with_warmup(optimizer, LR, epoch, warmup_epoch=20, scale=0.95, set_LR=0.001, interval_epoch=50):
    """Sets the learning rate to the initial LR decayed by 5% every interval epochs"""
    lr = LR
    if epoch < warmup_epoch:
        lr = np.around((set_LR/warmup_epoch) * (epoch+1), decimals=5)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr
        
    elif (epoch % interval_epoch) == 0:
        lr = np.around(LR * scale, decimals=5)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return lr

def inference(model, device, data_loader):
    model.eval()
    total_preds = torch.Tensor()
    with torch.no_grad():
        for data in tqdm(data_loader):
            data_mol = data[0].to(device)
            data_prot = data[1].to(device)
            pred = model(data_mol, data_prot)
            total_preds = torch.cat((total_preds, pred.cpu()), 0)
            """with open(result_train_name, 'a') as f:
                f.write("Drug id: "+ str(epoch+1) + ", LR : " + str(LR) + " ---> " + "Loss: " + str(train_loss) + " Trainning ci = " + str(train_ci) + '\n')"""
    return total_preds.numpy().flatten()

def save_checkpoint(state, filename = "states/checkpoint.pt"):
    torch.save(state, filename)

def resume(model, optimizer, savefile):
    if os.path.isfile(savefile):
        print("Loading checkpoint '{}'..".format(savefile))
        # checkpoint = torch.load(args.resume, map_location=device)
        checkpoint = torch.load(savefile)
        epoch = checkpoint['epoch']
        best_epoch = checkpoint['best_epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint['state_dict'])
        best_mse = checkpoint['best_mse']
        LR = checkpoint['LR']
        print("Checkpoint loaded . Resume training from epoch {}, LR = {}.".format(epoch, LR))
        return best_mse, best_epoch, epoch, optimizer, model, LR

# Drug protein pair for Training/Testing
def collate(data_list):
    batchA = Batch.from_data_list([data[0] for data in data_list])
    batchB = Batch.from_data_list([data[1] for data in data_list])
    return batchA, batchB
