# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 18:40:57 2023

@author: atver
"""
from torch import optim
import os 
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from VideoDataset_new import VideoDataset
from models_combined import vid_Model, sensor_Model, combineEarly, combineLate, train, validate, save_cm
from sklearn.utils import class_weight
from time_str import time_str
import wandb
from time import time
from sklearn.metrics import f1_score

# Greedy hyperparam optimization
# try optimizing one hyperparameter per CV fold
# 
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")
params = ["drop_out"] # ["optimizer", "batch_sz", "lrn_rate", "weight_decay", "drop_out"]
param_opts = {"optimizer":[optim.SGD, optim.Adagrad, optim.Adadelta, optim.Adam, optim.NAdam], 
           "batch_sz":[32,16,8],
           "lrn_rate":[.001,.0008,.0013,.0017],
           "weight_decay":[0, .001,.01,.05,.08], 
           "drop_out":[.1,.05,0]}
b_param = {"optimizer": optim.Adagrad, 
           "batch_sz":8,
           "lrn_rate":.0008, 
           "weight_decay":0, 
           "drop_out":0}
#save the best avg f1 score and save it to the first element results_param 
#so that it can be used to skip the training when evaluating the next set of options
model_classes = [combineEarly, combineLate]
model_class = combineLate
RL = False
save_models = False
SEG_TABLE_FN = "clips_30_100.csv" 
max_epochs = 30
#for ind in range(len(params)):
for param_i, param in enumerate(params): 
    optimizer= b_param["optimizer"] if (not isinstance(b_param["optimizer"],list)) else param_opts["optimizer"][0]
    lrn_rate = b_param["lrn_rate"] if not isinstance(b_param["lrn_rate"],list) else param_opts["lrn_rate"][0]
    weight_decay = b_param["weight_decay"] if not isinstance(b_param["weight_decay"],list) else param_opts["weight_decay"][0]
    batch_sz = b_param["batch_sz"] if not isinstance(b_param["batch_sz"],list) else param_opts["batch_sz"][0]
    drop_out = b_param["drop_out"] if not isinstance(b_param["drop_out"],list) else param_opts["drop_out"][0]
    
    if param_i==0:
        results_param = []
    for opt_i, opt in enumerate(param_opts[param]):
        if param_i!=0 and opt_i==0: # skip the trainings for a result we saw already
            continue
        locals()[param] = opt
        res_fold = [] 
        test_losses = []
        for fold in range(5): # Train each fold 
            print("fold ",str(fold))
            printonfold = "optimizer: {}\tlrn_rate: {}\tl2: {}\nbatch_sz: {}\tdropout: {}\tRL: {}\tmodel_class: {}".format(optimizer,lrn_rate,weight_decay,batch_sz,drop_out,RL,model_class)   
            print(printonfold)
            tstr = time_str() # get a time string to name temp files

            # Prepare DataLoaders
            vds_train = VideoDataset(SEG_TABLE_FN, group = "Train", fold = fold)
            vdl_train = DataLoader(vds_train, batch_size=batch_sz, shuffle=True, num_workers = 12, persistent_workers = 1, pin_memory=True)
            
            vds_valid = VideoDataset(SEG_TABLE_FN, group = "Valid", fold = fold)
            vdl_valid = DataLoader(vds_valid, batch_size=batch_sz, shuffle=False, num_workers = 12, persistent_workers = 1, pin_memory=True)
        
            # Get the class weights - make it a dictionary 
            class_weights = class_weight.compute_class_weight('balanced',
                                                          classes = np.array(['G0', 'G1', 'G2', 'G3', 'G4', 'G5'], dtype=object),
                                                          y = vds_train.clips_df["gesture"]) # should only be evaluated for train!
            
            # Set up Model    
            model = model_class(vds_train.clip_length, device, RL = RL, drop=drop_out)
            model.to(dtype = torch.float32, device = device)
            c_weight = torch.tensor(class_weights, dtype = torch.float32, device = device)
            loss_fn = torch.nn.CrossEntropyLoss(c_weight)
            
            optimizer_chosen = optimizer(model.parameters(), lr=lrn_rate, weight_decay = weight_decay)
        
            # Train Model
            early_stop = False
            f1_vs = []
            for t in range(max_epochs):
                print(f"Epoch {t+1}\n-------------------------------")
                if early_stop:
                    t = max_epochs-1
                start_train = time()
                model, acc_tr, train_avg_loss = train(vdl_train, model, loss_fn, optimizer_chosen, device) 
                end_train = time()
                # model0.eval() # set model to inference mode. This has been done in validate
                start_val = time()
                acc_v, val_avg_loss, y_list, pred_list = validate(vdl_valid, model, loss_fn, device, last = (t == (max_epochs-1)))
                end_val = time()
                f1_v = f1_score(y_list,pred_list,average='macro') # f1 macro of validation
                f1_vs.append(f1_v)
                try: #stop early when validation accuracy decreases or stays the same eight times in the last 10 epochs
                    if sum((np.array(f1_vs[-10:])-np.array(f1_vs[-11:-1]))<=0)>8:
                        print("STOPPING EARLY!")
                        early_stop = True 
                except ValueError:
                    print("except")
                    pass
                if t==max_epochs-1:
                    break
            res_fold.append(f1_v)
            print("res_fold: ", res_fold)
        results_param.append(sum(res_fold)/len(res_fold))
    best_opt_index = results_param.index(max(results_param))
    b_param[param] = param_opts[param][best_opt_index]
    print("/nBEST {} was {}".format(param, b_param[param]))
    print("param opts: ", param_opts[param])
    print("results_param: ", results_param,"\n")
    results_param = [max(results_param)]
    
    # can shift the list so that we are testing each param over all the folds
    #params = params[-1:] + params[:-1]
        

