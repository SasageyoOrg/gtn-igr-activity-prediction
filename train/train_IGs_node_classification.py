"""
    Utility functions for training one epoch 
    and evaluating one epoch
"""
import torch
import torch.nn as nn
import math
import dgl
from tqdm import tqdm
import time


from train.metrics import accuracy_MNIST_CIFAR as accuracy



def train_epoch(model, optimizer, device, data_loader, epoch):
  
    # print(f"Loading checkpoint")
    # checkpoint = torch.load("out\IGs\checkpoints\GraphTransformer_IG_GPU0_14h17m35s_on_Jul_12_2023\RUN_\epoch_106.pkl")
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # epoch = checkpoint['epoch']
    # loss = checkpoint['loss']

    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0
    gpu_mem = 0
    with tqdm(total=len(data_loader)) as t:
      for iter, (batch_graphs, batch_labels) in enumerate(data_loader):
          start = time.time()
          # t.set_description(f'- Epoch {epoch} -> batch {iter+1}')
          t.set_description(f'- Epoch {epoch}')
          
          batch_graphs = batch_graphs.to(device)
          batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
          batch_e = batch_graphs.edata['feat'].to(device)
          batch_labels = batch_labels.to(device)
          optimizer.zero_grad()
          try:
              batch_lap_pos_enc = batch_graphs.ndata['lap_pos_enc'].to(device)
              sign_flip = torch.rand(batch_lap_pos_enc.size(1)).to(device)
              sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
              batch_lap_pos_enc = batch_lap_pos_enc * sign_flip.unsqueeze(0)
          except:
              batch_lap_pos_enc = None
              
          try:
              batch_wl_pos_enc = batch_graphs.ndata['wl_pos_enc'].to(device)
          except:
              batch_wl_pos_enc = None

          batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_lap_pos_enc, batch_wl_pos_enc)
      
          loss = model.loss(batch_scores, batch_labels)
          loss.backward()
          optimizer.step()
          epoch_loss += loss.detach().item()
          epoch_train_acc += accuracy(batch_scores, batch_labels)
          nb_data += batch_labels.size(0)
          t.update()
      epoch_loss /= (iter + 1)
      epoch_train_acc /= nb_data
      t.set_postfix(time=time.time()-start,
                    train_loss=epoch_loss,
                    train_acc=epoch_train_acc)
      t.close()
    return epoch_loss, epoch_train_acc, optimizer, t


def evaluate_network(model, device, data_loader, epoch):
    
    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    nb_data = 0
    with torch.no_grad():
        for iter, (batch_graphs, batch_labels) in enumerate(data_loader):
            batch_graphs = batch_graphs.to(device)
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_e = batch_graphs.edata['feat'].to(device)
            batch_labels = batch_labels.to(device)
            try:
                batch_lap_pos_enc = batch_graphs.ndata['lap_pos_enc'].to(device)
            except:
                batch_lap_pos_enc = None
            
            try:
                batch_wl_pos_enc = batch_graphs.ndata['wl_pos_enc'].to(device)
            except:
                batch_wl_pos_enc = None
                
            batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_lap_pos_enc, batch_wl_pos_enc)
            loss = model.loss(batch_scores, batch_labels) 
            epoch_test_loss += loss.detach().item()
            epoch_test_acc += accuracy(batch_scores, batch_labels)
            nb_data += batch_labels.size(0)
        epoch_test_loss /= (iter + 1)
        epoch_test_acc /= nb_data
        
    return epoch_test_loss, epoch_test_acc


