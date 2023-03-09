# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 01:57:23 2023

@author: atver
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.metrics import f1_score

class vid_Model(nn.Module):
    def __init__(self, clip_len, device, RL=False, drop = 0):
        super().__init__()
        self.clip_len = clip_len
        self.mname = "vid_Model_" + str(self.clip_len)
        
        self.gru = nn.GRU(input_size=1280, hidden_size=8, batch_first=True)
        self.dense1 = nn.Linear(8*clip_len, 10)
        self.dense2 = nn.Linear(10, 6)
        self.device = device
        
    def forward(self, x):
        #print("shape x ", x.shape)
        vid_fts, sensor_fts = x[:,:,:1280], x[:,:,1280:]
        vid_fts = vid_fts.to(device = self.device, dtype = torch.float32)
        sensor_fts = torch.squeeze(sensor_fts)
        #print("shape vid_fts ", vid_fts.shape)
        #print("shape sensor_fts ", sensor_fts.shape)
        #vid_fts, sensor_fts = torch.split(x, [1280,36],dim=1) # dim0 = batch
        # y = torch.permute(vid_fts,(0,2,1))
        y, _ = self.gru(vid_fts) #
        y = y.reshape(-1, self.gru.hidden_size*(1+self.gru.bidirectional)*self.clip_len) # Reshape
        y = self.dense1(y)
        y = self.dense2(y)
        #print("shape y ", y.shape)
        # y = self.softmax(y) # Present in cross entropy loss already
        return y

# model = vid_finalstate_Model()

class sensor_Model(nn.Module):
    def __init__(self, clip_len, device, RL=False, drop=0):
        super().__init__()
        self.clip_len = clip_len
        self.mname = "sensor_Model_" + str(self.clip_len)
        self.gru = nn.GRU(input_size=36, hidden_size=8, batch_first=True)
        self.dense1 = nn.Linear(8*clip_len, 10)
        self.dense2 = nn.Linear(10, 6)
        self.device = device
        
    def forward(self, x):
        vid_fts, sensor_fts = x[:,:,:1280], x[:,:,1280:]
        vid_fts = vid_fts.to(device = self.device, dtype = torch.float32)
        sensor_fts = sensor_fts.to(device = self.device, dtype = torch.float32)
        y, _ = self.gru(sensor_fts)
        y = y.reshape(-1, self.gru.hidden_size*(1+self.gru.bidirectional)*self.clip_len) # Reshape
        y = self.dense1(y)
        y = self.dense2(y)
        return y


class combineEarly(nn.Module):

    def __init__(self, clip_len, device, RL = False, drop = 0):
        super().__init__()  
        self.mname = "combineEarly_" + str(clip_len)
        if RL:
            self.mname = "RL_combineEarly_" + str(clip_len)
        NUM_FEATURES = 1280 + 36  # frame features + 36 kinematic
        RNN_OUT = 64 #32
        NUM_TARGETS = 6
        bi = True
        self.clip_len = clip_len
        self.device = device
        self.RL = RL
        if RL:
            self.dense0 = nn.Linear(NUM_FEATURES,NUM_FEATURES)
            self.leaky0 = nn.LeakyReLU()
        self.lstm = nn.LSTM(input_size=NUM_FEATURES, hidden_size=RNN_OUT, bidirectional=bi, batch_first=True)
        self.dropout = torch.nn.Dropout(p=drop)
        self.dense1 = nn.Linear((1+bi)*RNN_OUT,128,device=device) #(2*(1+bi)*RNN_OUT)
        self.leaky1 = nn.LeakyReLU()
        self.dense2 = nn.Linear(128,64,device=device)
        self.tanh2 = nn.Tanh()
        self.dense3 = nn.Linear(64,NUM_TARGETS,device=device)
        self.leaky3 = nn.LeakyReLU()
    
    
    def forward(self, x):
        fts = x.to(device = self.device, dtype = torch.float32)
        if self.RL:
            fts = self.dense0(fts)
            fts = self.leaky0(fts)
        # print("fts ", fts.shape)
        _, (final_bft, _) = self.lstm(fts)
        # print("final_bft ", final_bft.shape) # (D*num_layers, N, Hout)   [2*1, batch12, 64]
        final_bft = torch.permute(final_bft, (1,0,2)) # final_bft = N,D∗num_layers,H_out
        final_bft = final_bft.reshape(final_bft.shape[0], -1) # final_v = N,D∗num_layers*H_out
        # print("final_bft ", final_bft.shape) # final_v = N,D∗num_layers*H_out
        output = self.dropout(final_bft)
        output = self.dense1(output)
        output = self.leaky1(output)
        # print("output1 ", output.shape)
        output = self.dense2(output)
        output = self.tanh2(output)   
        # print("output2 ", output.shape)
        output = self.dense3(output)
        output = self.leaky3(output)
        #print("output3 ", output.shape)
        return output
    
    
class combineLate(nn.Module):
    def __init__(self, clip_len, device, RL=False, drop = 0):
        super().__init__()
        self.mname = "combineLate_" + str(clip_len)
        if RL:
            self.mname = "RL_combineLate_" + str(clip_len)
        bi = True
        NUM_FEATURES_VIDEO = 1280
        NUM_FEATURES_KINEMATIC = 36
        NUM_TARGETS = 6
        RNN_OUT = 32
        self.clip_len = clip_len
        self.device = device
        self.RL = RL
        if RL:
            self.dense0v = nn.Linear(NUM_FEATURES_VIDEO,NUM_FEATURES_VIDEO)
            self.leaky0v = nn.LeakyReLU()
            self.dense0k = nn.Linear(NUM_FEATURES_KINEMATIC,NUM_FEATURES_KINEMATIC)
            self.leaky0k = nn.LeakyReLU()            
        self.lstm_video =     nn.LSTM(input_size=NUM_FEATURES_VIDEO,     hidden_size=RNN_OUT, batch_first=True, bidirectional=bi)# only when num_layer>1 dropout = .2
        self.lstm_kinematic = nn.LSTM(input_size=NUM_FEATURES_KINEMATIC, hidden_size=RNN_OUT, batch_first=True, bidirectional=bi)
        self.dropoutv = torch.nn.Dropout(p=drop)
        self.dropoutk = torch.nn.Dropout(p=drop)
        self.dense1 = nn.Linear(2*(1+bi)*RNN_OUT,128,device=device) #(2*(1+bi)*RNN_OUT)
        self.leaky1 = nn.LeakyReLU()
        self.dense2 = nn.Linear(128,64,device=device)
        self.tanh2 = nn.Tanh()
        self.dense3 = nn.Linear(64,NUM_TARGETS,device=device)
        self.leaky3 = nn.LeakyReLU()
    
    
    def forward(self, x):
        vid_fts, sensor_fts = x[:,:,:1280], x[:,:,1280:]

        vid_fts = vid_fts.to(device = self.device, dtype = torch.float32)
        sensor_fts = sensor_fts.to(device = self.device, dtype = torch.float32)
        #print("vid_fts ",vid_fts.shape)
        #print("sensor_fts ", sensor_fts.shape)
        if self.RL:
            vid_fts = self.dense0v(vid_fts)
            vid_fts = self.leaky0v(vid_fts)
            sensor_fts = self.dense0k(sensor_fts)
            sensor_fts = self.leaky0k(sensor_fts)
        
        _, (final_v, _) = self.lstm_video(vid_fts)
        _, (final_s, _) = self.lstm_kinematic(sensor_fts)
        
        #print("final_v ", final_v.shape) # (D*num_layers, N, Hout)
        #print("final_s ", final_s.shape) # [2*1, batch12, 32]
        final_v = torch.permute(final_v, (1,0,2)) # final_v = N,D∗num_layers,H_out
        final_s = torch.permute(final_s, (1,0,2))
        final_v = final_v.reshape(final_v.shape[0], -1) # final_v = N,D∗num_layers*H_out
        final_s = final_s.reshape(final_s.shape[0], -1)
        #print("final_v ", final_v.shape) # (D*num_layers, N, Hout)
        #print("final_s ", final_s.shape) # [2*1, batch12, 32]
        final_v = self.dropoutv(final_v)
        final_s = self.dropoutk(final_s)
        combined_tensor = torch.concat([final_v, final_s], dim=1).float()
        #print("combined_tensor ", combined_tensor.shape)
        output = self.dense1(combined_tensor)
        output = self.leaky1(output)
        #print("output1 ", output.shape)
        output = self.dense2(output)
        output = self.tanh2(output)   
        #print("output2 ", output.shape)
        output = self.dense3(output)
        output = self.leaky3(output)
        #print("output3 ", output.shape)
        return output



def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    avgb_loss, correct = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device, dtype = torch.float32), y.argmax(1).to(device, dtype = torch.long)
        y=torch.reshape(y, (-1,))
        # Compute prediction error
        pred = model(X)
        pred = pred.to(device, dtype = torch.float32)
        # print("pred type ", pred.dtype)
        # print("y(target) shpae and type ", y.shape, y.dtype)
        loss = loss_fn(pred, y) # target for cross entropy is the index
        avgb_loss += loss.item()
        correct += (pred.argmax(1) == y).sum().item()

        # Backpropagation
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        #if batch % 100 == 0:
            #loss, current = loss.item(), (batch + 1) * len(X)
            # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    # Return the metrics for epoch to be logged    Top k categorical?   
    avgb_loss /= num_batches
    correct /= size 
    print(f"Train Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {avgb_loss:>8f} \n")
    return model, correct, avgb_loss
    
    
def validate(dataloader, model, loss_fn, device, last=0):
    # used for validation and test
    test_mode = False
    if dataloader.dataset.group.lower() == "test":
        test_mode = True
    with torch.no_grad():
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, correct = 0, 0
        batch = 0
        pred_list = []
        y_list = []
        for X, y in dataloader:
            X, y = X.to(device, dtype = torch.float32), y.argmax(1).to(device, dtype = torch.long)
            y=torch.reshape(y, (-1,))
            # Compute prediction error
            pred = model(X)
            pred = pred.to(device, dtype = torch.float32)
            # if test_mode or last:
            pred_list += pred.argmax(1).tolist()
            y_list += y.tolist()
            try:
                test_loss += loss_fn(pred.to(device, dtype = torch.float32), y).item()
            except:
                print("in batch# ", batch)
                print("validate pred size ", pred.shape)
                print("validate target y size ", y.shape)
                print("fold ",dataloader.dataset.fold)
                print("size alllabs ", dataloader.dataset.alllabs.shape)
                print("size allvids ", dataloader.dataset.allvids.shape)
                print("size allsens ", dataloader.dataset.allsens.shape)
                print("problem batch y: ", y[dataloader.batch_size*batch:])
            correct += (pred.argmax(1) == y).sum().item()
            batch += 1
    test_loss /= num_batches
    correct /= size
    f1_v = f1_score(y_list,pred_list,average='macro') # f1 macro of validation
    if test_mode:
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}, F1 score: {f1_v:>8f}\n")
    else:
        print(f"Validation Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}, F1 score: {f1_v:>8f}\n")        
    if test_mode or last:
        save_name = "cm_{}_{}_fold{}.png".format(model.mname,dataloader.dataset.group,dataloader.dataset.fold)
        save_cm(y_list, pred_list, save_name)

    return correct, test_loss, y_list, pred_list
    
def save_cm(y_list, pred_list, save_name): 
    ConfusionMatrixDisplay.from_predictions(y_list, pred_list, normalize = "true", display_labels=["G"+str(i) for i in range(6)], cmap = mpl.colormaps['RdYlGn'])
    plt.title(save_name)
    plt.savefig(save_name)
    plt.close()

    

