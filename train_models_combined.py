# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 00:58:17 2023

@author: atver
"""
# train with the train set
# hyperparameter tune with validation sets

# after selection of hyper parameters evaluate the five models, with the best hyperparams trained on just the train, on the test sets

# Pytorch model using final state of GRU
# TODO redo the groups by architecture
import os 
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from VideoDataset_new import VideoDataset
from models_combined import vid_Model, sensor_Model, combineEarly, combineLate, train, validate
from sklearn.utils import class_weight
from time_str import time_str
import wandb
from time import time
from sklearn.metrics import f1_score


# TODO having 2 dropouts for the 2 streams maybe bad since the data loss is uncorraltaed?
# an entry in csv 60_100 was removed from vid  P024_balloon1 first  6699 last  6758
save_models = True
SEG_TABLE_FNs = ["clips_30_100.csv"] #,"clips_60_100.csv"]

weight_decays = [0] #, .1, .2,.5]
lrn_rates = [.008] #.001,.005,.0005]
dropouts = [0]#, .05, .1, .2]
RLs = [False] #, True]
model_classes = [combineEarly, combineLate]
withWANDB = True
WANDBonline = 'offline'
test_mode = True
# include dropout torch.nn.Dropout a good spot would be at the output of the RNN

for SEG_TABLE_FN in SEG_TABLE_FNs:
    for weight_decay in weight_decays:
        for lrn_rate in lrn_rates:
            for drop in dropouts:
                for RL in RLs:
                    for model_class in model_classes:   
                    
                        #for weight_decay in weight_decays:
                        printonfold = "table: {}\nl2: {}\nlrn_rate: {}\nRL: {}\nmodel_class: {}\ndropout: {}".format(SEG_TABLE_FN,weight_decay,lrn_rate,RL,model_class,drop)   
                        
                        BATCH_SZ = 8
                        EPOCHS = 30
                        #SEG_TABLE_FN = "clips_30_100.csv"
                        #lrn_rate = 0.001
                        tstr = time_str() # get a time string to name temp files
                        #model_class = vid_finalstate_Model
                        #RL = True
                        
                        # wanb log param
                        wandb_project = "cvsa_final_project"
                        wandb_config = {
                            "learning_rate": lrn_rate,
                            "architecture": "LSTM",
                            "feature_streams": "vid", # "together","sep","sensor"
                            "dataset": SEG_TABLE_FN,
                            "epochs": EPOCHS,
                            "weight_decay": weight_decay,
                            "RL": RL,
                            "drop_out": drop,
                            "fresh": True
                            }
                        if model_class in [combineEarly, combineLate]:
                            wandb_config["architecture"] = "LSTM"
                        if model_class == combineEarly:
                            wandb_config["feature_streams"] = "together"
                        if model_class == combineLate:
                            wandb_config["feature_streams"] = "sep"
                            
                        wandb_save_code = True
                        wandb_notes = ""
                        
                        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
                        print(f"Using {device} device")
                        
                        test_losses = []
                        for fold in range(5):
                            print("fold ",str(fold))
                            print(printonfold)
                            # Prepare DataLoaders
                            vds_train = VideoDataset(SEG_TABLE_FN, group = "Train", fold = fold)
                            vdl_train = DataLoader(vds_train, batch_size=BATCH_SZ, shuffle=True, num_workers = 12, persistent_workers = 1, pin_memory=True)
                            
                            #vds_valid = VideoDataset(SEG_TABLE_FN, group = "Valid", fold = fold)
                            #vdl_valid = DataLoader(vds_valid, batch_size=BATCH_SZ, shuffle=False, num_workers = 12, persistent_workers = 1, pin_memory=True)
                            
                            # Performance on test sets will be evaluated once we have chosen our hyperparameters
                            vds_test  = VideoDataset(SEG_TABLE_FN, group = "Test",  fold = fold)
                            vdl_test  = DataLoader(vds_test, batch_size=BATCH_SZ, shuffle=False, num_workers = 12, persistent_workers = 1, pin_memory=True)
                        
                            # Get the class weights - make it a dictionary so that it can be used in model.fit
                            class_weights = class_weight.compute_class_weight('balanced',
                                                                             classes = np.array(['G0', 'G1', 'G2', 'G3', 'G4', 'G5'], dtype=object),
                                                                             y = vds_train.clips_df["gesture"]) # should only be evaluated for train!
                            
                        
                            # Set up Model    
                            model0 = model_class(vds_train.clip_length, device, RL = RL, drop=drop)
                            model0.to(dtype = torch.float32, device = device)
                            c_weight = torch.tensor(class_weights, dtype = torch.float32, device = device)
                            loss_fn = torch.nn.CrossEntropyLoss(c_weight)
                            
                            optimizer = torch.optim.SGD(model0.parameters(), lr=lrn_rate, weight_decay = weight_decay)
                        
                            # Prepare WandB log for training
                            # start a new wandb run to track this script
                            os.environ['WANDB_MODE'] = WANDBonline # hopefully faster
                            wandb.init(
                                # group 
                                group = model0.mname +" Fold "+ str(fold)+"_Final1", 
                                #job_type
                                job_type = "Train",
                                # set the wandb project where this run will be logged
                                project= wandb_project,
                                # track hyperparameters and run metadata
                                config = wandb_config,
                                save_code = wandb_save_code,
                                notes = wandb_notes)
                        
                            # Train Model
                            early_stop = False
                            f1_vs = []
                            for t in range(EPOCHS):
                                if early_stop:
                                    t=EPOCHS-1
                                if not early_stop:
                                    print(f"Epoch {t+1}\n-------------------------------")
                                    start_train = time()
                                    model0, acc_tr, train_avg_loss = train(vdl_train, model0, loss_fn, optimizer, device) 
                                    end_train = time()
                                    # model0.eval() # set model to inference mode. This has been done in validate
                                    start_val = time()
                                    acc_v, val_avg_loss, y_list, pred_list = validate(vdl_test, model0, loss_fn, device, t ==EPOCHS-1)
                                    end_val = time()
                                    f1_v = f1_score(y_list,pred_list,average='macro') # f1 macro of validation
                                    f1_vs.append(f1_v)
                                    # log metrics to wandb
                                    wandb.log({"train_avg_loss": train_avg_loss,
                                    "train_accuracy": acc_tr,
                                    "test_avg_loss": val_avg_loss,
                                    "test_accuracy": acc_v,
                                    "train_time": round((end_train - start_train) / 60, 2),
                                    "test_time": round((end_val - start_val) / 60, 2),
                                    "test_f1_score": f1_v})  
                                    f1_vs.append(f1_v)
                                    try: #stop early when validation accuracy decreases eight times in the last 10 epochs
                                        if sum((np.array(f1_vs[-10:])-np.array(f1_vs[-11:-1]))<=0)>8 and not test_mode:
                                            print("STOPPING EARLY!")
                                            early_stop = True 
                                    except ValueError:
                                        pass
                                else:
                                    t = EPOCHS-1
                                    
                                
                                # save model on the epoch 
                                if save_models:
                                    try: 
                                        MODEL_PATH = "{}_epochs{}_{}.pth".format(tstr,t+1,model0.mname)
                                        if t != EPOCHS-1:
                                            torch.save(model0.state_dict(), MODEL_PATH)
                                        if t == EPOCHS-1:
                                            MODEL_PATH = "{}_{}epochs.pth".format(model0.mname,EPOCHS)
                                            torch.save(model0.state_dict(), MODEL_PATH)
                                        if t != 0 and t != EPOCHS-1:
                                            MODEL_PATH = "{}_epochs{}_{}.pth".format(tstr,t,model0.mname)
                                            os.remove(MODEL_PATH)
                                    except:
                                        pass
                                if t==(EPOCHS-1):
                                    break
                                    
                                    
                            # [optional] finish the wandb run, necessary in notebooks
                            wandb.finish()
                            
                            # load the model
                            #model0 = model_class(vds_train.clip_length, device)
                            #model0.load_state_dict(torch.load(MODEL_PATH))
            
          
                
        
        
        
        
        
    
    
    