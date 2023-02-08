from torch_geometric.loader import DataLoader
from core.data_process_1DGCN import prepare_dataset_withFolds
from core.utils import *
from core.emetrics import *
from core.models.GCNNet import GCNNet, GCNEdgeNet
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import sys

import config 

def GET_CONFIG():
    filename = config.filename
    dataset = config.dataset
    BATCH_SIZE = config.BATCH_SIZE
    NUM_EPOCHS = config.NUM_EPOCHS
    MAX_LR = config.max_lr
    LR = config.lr
    FOLD = config.FOLD
    RESUME = config.RESUME_TRAIN
    WINDOWS = config.windows

    print('Filename: ', filename)
    print('Training from checkpoint : ', str(RESUME), '\n')
    print('Setting Epochs: ', NUM_EPOCHS)
    print('Setting Learning rate: ', LR)
    print('Batch size: ', BATCH_SIZE)
    print('Graph windows: ', WINDOWS)
    print('\n')
    return filename, dataset, BATCH_SIZE, NUM_EPOCHS, MAX_LR, LR, FOLD, RESUME


def GET_MODEL():
    model = GCNEdgeNet(num_features_xd=66, num_features_xt=1280, dropout=0, edge_input_dim=18)
    model_st = GCNEdgeNet.__name__
    #model = GCNNet(num_features_xd=66, num_features_xt=1280, dropout=0)
    #model_st = GCNNet.__name__
    #model = RGCNNet(num_features_xd=66, num_features_xt=1280, dropout=0)
    #model_st = RGCNNet.__name__
    return model, model_st


def SAVE_LOCATION(fname, models_dir = 'models', results_dir = 'results', states_dir = 'states', figure_dir = 'figures'):
    models_dir = models_dir
    results_dir = results_dir
    states_dir = states_dir
    figure_dir = figure_dir

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    if not os.path.exists(states_dir):
        os.makedirs(states_dir)
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)

    model_file_name = models_dir + '/model_' + fname
    figure_file_name = figure_dir + '/fig_' + fname
    model_state_name = states_dir + '/checkpoint_' + fname + '.model'
    result_train_name = results_dir + '/trainlog_' + fname  + '.txt'
    result_eval_name = results_dir + '/validlog_' + fname  + '.txt'

    return model_file_name, model_state_name, result_train_name, result_eval_name, figure_file_name


def LOAD_DATA(train=None, valid=None, test=None, batch_size=128):
    train_loader, valid_loader, test_loader = None, None, None
    if train != None:
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=collate, pin_memory=True, num_workers=4)
    if valid != None:
        valid_loader = DataLoader(valid, batch_size=batch_size, shuffle=False, collate_fn=collate, pin_memory=True, num_workers=4)
    if test != None:
        test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, collate_fn=collate)
    return train_loader, valid_loader, test_loader


def GET_RESUME_PARAMS(models, optimizers, model_states):
    best_mse, best_epoch, start_epoch, optimizer, model, LR = resume(model=models, optimizer=optimizers, savefile=model_states)
    print("LOADING CHECKPOINT COMPLETE . . : best_mse = ", str(best_mse), "\n")
    print("STARTING AT EPOCH : ", str(start_epoch), "\n")
    return best_mse, best_epoch, start_epoch, optimizer, model, LR

    
def PLOT_LOSS(history, fignames, start_epoch=None):
    start = 0 if start_epoch==None else start_epoch
    figname = fignames 
    plt.figure()
    plt.plot(history['train_loss'][start:])
    plt.plot(history['valid_loss'][start:])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['trainloss', 'valloss'], loc='upper left')
    plt.savefig(figname + "_loss" + ".png" , dpi=None, facecolor='w', edgecolor='w', orientation='portrait', 
                    format=None,transparent=False, bbox_inches=None, pad_inches=0.1)
    plt.close()

    ## PLOT CINDEX
    plt.figure()
    plt.title('model concordance index')
    plt.ylabel('cindex')
    plt.xlabel('epoch')
    plt.plot(history['train_ci'][start:])
    plt.plot(history['valid_ci'][start:])
    plt.legend(['traincindex', 'valcindex'], loc='upper left')
    plt.savefig(figname + "_ci" + ".png" , dpi=None, facecolor='w', edgecolor='w', orientation='portrait', 
                            format=None,transparent=False, bbox_inches=None, pad_inches=0.1)
    plt.close()

def RUN_DTA(train_data, valid_data, device):
    filename, dataset, BATCH_SIZE, NUM_EPOCHS, MAX_LR, LR, FOLD, RESUME = GET_CONFIG()
    model_file_name, model_state_name, result_train_name, result_eval_name, figure_file_name = SAVE_LOCATION(fname=filename)

    model, model_st = GET_MODEL()
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.02)
    criterion = torch.nn.MSELoss()

    train_loader, valid_loader, _ = LOAD_DATA(train=train_data, valid=valid_data, test=None, batch_size=BATCH_SIZE)

    best_mse = 1000
    best_ci = 0
    best_epoch = -1
    start_epoch = 0

    train_loss_list = []
    valid_loss_list = []
    train_ci_list = []
    valid_ci_list = []
    # LOAD MODEL PARAMS IF RESUME
    if RESUME == True:
        best_mse, best_epoch, start_epoch, optimizer, model, LR = GET_RESUME_PARAMS(models=model, optimizers=optimizer, model_states=model_state_name)
    
    print("Training . . . .\n")
    for epoch in range(start_epoch, NUM_EPOCHS):
        train_loss, train_ci = train(model, device, train_loader, optimizer, criterion, epoch+1)
        ground_truth, prediction = evaluate(model, device, valid_loader)
        metric = [get_mse(ground_truth, prediction), get_ci(ground_truth, prediction)]
        train_loss_list.append(np.around(train_loss, decimals=4))
        valid_loss_list.append(np.around(metric[0], decimals=4))
        train_ci_list.append(np.around(train_ci, decimals=4))
        valid_ci_list.append(np.around(metric[1], decimals=4))
        LR = LR_scheduler_with_warmup(optimizer, LR, epoch, warmup_epoch=50, scale=0.8, set_LR=MAX_LR, interval_epoch=100)

        with open(result_train_name, 'a') as f:
            f.write("On epoch: "+ str(epoch+1) + ", LR : " + str(LR) + " ---> " + "Loss: " + str(train_loss) + " Trainning ci = " + str(train_ci) + '\n')

        with open(result_eval_name, 'a') as f:
            f.write("On epoch" + str(epoch+1) + ", Validation mse --> " + str(metric[0]) + ", Validation ci --> " + str(metric[1]) + '\n')
      
        # If select model by ci score.
        """if metric[1] > best_ci:
            best_ci = metric[1]
            best_epoch = epoch+1
            print("On epoch", best_epoch, ", Validation error decrease to --> ", str(metric[0]), ", CI score --> ", str(best_ci), " Select by ci")
            torch.save(model.state_dict(), model_file_name + '.model') # Save best perform on validation set"""

        if metric[0] < best_mse:
            best_mse = metric[0]
            best_epoch = epoch+1
            print("On epoch", best_epoch, ", Validation error decrease to --> ", best_mse, ", CI score --> ", str(metric[1]))
            torch.save(model.state_dict(), model_file_name + '.model') # Save best perform on validation set

    history = {"train_loss": train_loss_list, "train_ci":train_ci_list, "valid_loss":valid_loss_list, "valid_ci":valid_ci_list}
    PLOT_LOSS(history=history, fignames=figure_file_name, start_epoch=3)
    torch.save(model.state_dict(), model_file_name + '_finale.model') # Save Final model (Last Epoch)

    states = {'epoch': epoch + 1,
            'best_epoch': best_epoch,
            'model_name': model_st,
            'state_dict': model.state_dict(),
            'best_mse': best_mse,
            'optimizer': optimizer.state_dict(),
            'LR':LR}

    save_checkpoint(state=states, filename=model_state_name)
    print("\n. . . Training Complete . . .\n")
    


def EVAL_TEST(test_data, model_list, device):
    """
        This function evaluate model that train on 4-fold training set & selected by validation set.
        [train][train][train][train][validation] --> (model state dict)
        then make an evaluation on hold out test set (independence test set) by using above learned model.
    """
    model, _ = GET_MODEL()
    model = model.to(device)
    model_file_name, _, _, result_eval_name, _ = SAVE_LOCATION(fname=config.filename)
    _, _, test_loader = LOAD_DATA(train=None, valid=None, test=test_data, batch_size=config.BATCH_SIZE)

    for m in model_list:
    # Test set prediction
        try:
            model.load_state_dict(torch.load(model_file_name+m))
        except:
            raise Exception(m + " file does not exists . .")

        print("Testing on test set . . . Model --> ", m, "\n")
        ground_truth, prediction = evaluate(model, device, test_loader)
        
        metric = [get_mse(ground_truth, prediction),
                get_rmse(ground_truth, prediction),
                get_ci(ground_truth, prediction),
                get_pearson(ground_truth, prediction),
                get_spearman(ground_truth, prediction)]

        print("Test set MSE : ", metric[0])
        print("Test set rmse : ", metric[1])
        print("Test set CI score : ", metric[2])
        print("Test set pearson : ", metric[3])
        print("Test set spearman : ", metric[4])
        print("\n")
        with open(result_eval_name, 'a') as f:
            f.write("\nTest set evaluation: "+ str(m) + '\n')
            f.write("Test set MSE : "+ str(metric[0]) + '\n')
            f.write("Test set rmse : "+ str(metric[1]) + '\n')
            f.write("Test set CI score : "+ str(metric[2]) + '\n')
            f.write("Test set pearson : "+ str(metric[3]) + '\n')
            f.write("Test set spearman : "+ str(metric[4]) + '\n')


if __name__=="__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Train on: ', device, '\n')

    train_data, valid_data, test_data = prepare_dataset_withFolds(dataset=config.dataset, path="dataset/", fold=config.FOLD, windows=config.windows)
     
    #Deprecated
    #IF no fold, valid_data and test_data are the same split. This is just coding simplicity sake.
    #train_data, valid_data, test_data = prepare_dataset(dataset=config.dataset, windows=config.windows)
   
    RUN_DTA(train_data=train_data, valid_data=valid_data, device=device)
    test_list = [".model", "_finale.model"]
    EVAL_TEST(test_data=test_data, model_list=test_list, device=device)
