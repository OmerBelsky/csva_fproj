# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 19:47:12 2023

@author: atver
"""
import os 
import pandas as pd
import numpy as np
import torch
import pickle
from torch.utils.data import Dataset
from prepare_datanpys import prepare_datanpys

class VideoDataset(Dataset):
    def __init__(self, clips_df, fold=0, group="None"):
        # clips_df is a string specifying the file name for the table of segments
        # group None Train Test Valid
        # fold the current fold in the cross validation        
        SEG_TABLE_FN = clips_df
        self.fold = fold  
        #norm_consts = pd.read_csv("sensor_velocity_statistics_fold_{}.csv".format(self.fold))
        #self.means = np.expand_dims(np.array(norm_consts.iloc[1][1:]),axis=1)
        #self.stds  = np.expand_dims(np.array(norm_consts.iloc[2][1:]),axis=1)
        if type(clips_df) == str:
            SEG_TABLE_FN = SEG_TABLE_FN + ".csv" if ".csv" not in SEG_TABLE_FN else SEG_TABLE_FN
            clips_df = pd.read_csv(SEG_TABLE_FN)
            folds_path = r"/datashare/APAS/folds"
            test_videos = []
            with open(os.path.join(folds_path,"test "+ str(fold) +".txt")) as f:
                for ln in f: # for each line
                    test_videos += [ln.split()[0][:-4]] 
            valid_videos = []
            with open(os.path.join(folds_path,"valid "+ str(fold) +".txt")) as f:
                for ln in f: # for each line
                    valid_videos += [ln.split()[0][:-4]]
            if group.lower() == "none":
                assert 0, "A group should be specified if clips_df is a file name"
            elif group.lower() == "train":
                clips_df = clips_df[~clips_df['video fn'].isin(test_videos + valid_videos)]
            elif group.lower() == "valid":
                clips_df = clips_df[clips_df['video fn'].isin(valid_videos)]
            elif group.lower() == "test":
                clips_df = clips_df[clips_df['video fn'].isin(test_videos)]
                
        # I had to add this line in beacause we are missing P031 in video features. 
        clips_df = clips_df[~clips_df['video fn'].str.contains("31")]
        
        self.clips_df = clips_df 
        self.clip_length = clips_df["last"].iloc[0] - clips_df["first"].iloc[0] + 1
        
        cats = ["G0","G1","G2","G3","G4", "G5"]
        self.onehot_dict = {gest:[0]*i + [1] + [0]*(len(cats)-i-1) for i, gest in enumerate(cats)}
        self.group = group
        
        vidftspath =    SEG_TABLE_FN[6:-4] + "_vid_fold" + str(fold) + "_" + group.lower() + ".pickle"
        sensorftspath = SEG_TABLE_FN[6:-4]+"_sen_fold"+str(fold)+"_"+group.lower()+".pickle"
        labelpath     = SEG_TABLE_FN[6:-4]+"_lab_fold"+str(fold)+"_"+group.lower()+".pickle"
        prepare_datanpys(fold=fold,SEG_TABLE_FN=SEG_TABLE_FN,group=group)
        with open("ds_npys/"+vidftspath,"rb") as f:  
            self.allvids = pickle.load(f)
        with open("ds_npys/"+sensorftspath,"rb") as f:  
            self.allsens = pickle.load(f)
        with open("ds_npys/"+labelpath,"rb") as f:  
            self.alllabs = pickle.load(f)

        

    
    def __len__(self): 
        # print("length of "+ self.group + " clip_df: ", len(self.clips_df))
        return len(self.clips_df)
    
    def __getitem__(self, index):
        # TODO add a try statement if error I want to see which! one
        batch_clips = np.concatenate((self.allvids[index,:,:], self.allsens[index,:,:]),axis=1)
        batch_label = self.alllabs[index,:]
        try:
            assert(type(batch_clips)==np.ndarray)
            assert(type(batch_label)==np.ndarray)
            assert(batch_clips.shape==(30,1316))
            assert(batch_label.shape==(6,))
        except:
            print("batch_clips type ", type(batch_clips))
            print("batch_clips shape ",batch_clips.shape)
            print("batch_label type ", type(batch_label))
            print("batch_label shape ",batch_label.shape)
        batch_clips=np.vstack(batch_clips).astype(np.float)
        batch_label=np.vstack(batch_label).astype(np.float)
        
        batch_clips = torch.tensor(batch_clips, dtype = torch.float32)
        batch_label= torch.tensor(batch_label, dtype = torch.float32)
        return batch_clips, batch_label
            
    # def _extract_vidft_clip(self,i):
    #     clip = np.load(r"/datashare/APAS/features/fold" + str(self.fold)+r"/" +self.clips_df["video fn"].iloc[i] +".npy", 
    #                    mmap_mode = "r")[:,self.clips_df["first"].iloc[i]:self.clips_df["last"].iloc[i]+1]
    #     return np.array(clip).astype('float64')
    
    # def _extract_sensorft_clip(self,i):
    #     clip = np.load(r"/datashare/APAS/kinematics_npy/" +self.clips_df["video fn"].iloc[i] +".npy", 
    #                    mmap_mode = "r")[:,self.clips_df["first"].iloc[i]:self.clips_df["last"].iloc[i]+1]
    #     clip = self._transform_sensor(np.array(clip))
    #     return np.array(clip).astype('float64')

    # def _get_label(self,i):
    #     # code to extract label from video path
    #     label = self.clips_df['gesture'].iloc[i]
    #     label = self.onehot_dict[label]
    #     return np.array(label).astype('float64')

    # def _transform_sensor(self,clip):
    #     # adjust over angle velocity
    #     #for ang_set in range(6):
    #     #    angles = clip[3+ang_set*6:6+ang_set*6,:]
    #     #    angles[angles>  180] -= 360
    #     #    angles[angles< -180] += 360
    #     #    clip[3+ang_set*6:6+ang_set*6,:] = angles 
    #     # I can directly appy this since the pos velocity is never very large. Will speed things up :)
    #     clip[clip> 180] -= 360
    #     clip[clip<-180] += 360
    #     # broadcast the normalization clip.shape = (36,timesteps) means.shape =(36,1)      
    #     clip = (clip-self.means)/self.stds    
    #     return np.array(clip).astype('float64')
    
    

