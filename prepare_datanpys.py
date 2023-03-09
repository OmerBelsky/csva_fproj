# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 13:53:57 2023

@author: atver
"""
# 30_100_fold0_train only one for sensor

# for train valid and test separate for sensor and vid
# build a numpy where rows are a single clip
# index of clip
# then pickle it

# this will be read in in dataset

import os 
import pandas as pd
import numpy as np
import pickle
from os import listdir
from os.path import isfile, join



def prepare_datanpys(fold,SEG_TABLE_FN,group):
    # clear ds_npys if none of the files are for the specified seg_table_fn
    onlyfiles = [f for f in listdir("ds_npys/") if isfile(join("ds_npys/", f))]
    if not any([True if SEG_TABLE_FN[6:-4] in f else False for f in onlyfiles]):
        for f in onlyfiles:
            os.remove(join("ds_npys/", f))
    # npy files with clips names
    vidftspath =    SEG_TABLE_FN[6:-4]+"_vid_fold"+str(fold)+"_"+group.lower()+".pickle"
    sensorftspath = SEG_TABLE_FN[6:-4]+"_sen_fold"+str(fold)+"_"+group.lower()+".pickle"
    labelpath     = SEG_TABLE_FN[6:-4]+"_lab_fold"+str(fold)+"_"+group.lower()+".pickle"
    if not os.path.exists("ds_npys/"+vidftspath) or not os.path.exists("ds_npys/"+sensorftspath) or not os.path.exists("ds_npys/"+labelpath):
        if type(SEG_TABLE_FN) == str:
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
            clip_length = clips_df["last"].iloc[0] - clips_df["first"].iloc[0] + 1
            
            cats = ["G0","G1","G2","G3","G4", "G5"]
            onehot_dict = {gest:[0]*i + [1] + [0]*(len(cats)-i-1) for i, gest in enumerate(cats)}
            allvid_clips_data = np.zeros((len(clips_df),clip_length,1280))
            allsens_clips_data = np.zeros((len(clips_df),clip_length,36))
            alllab_clips_data = np.zeros((len(clips_df),6))
            clipi = 0
            for vid in clips_df['video fn'].unique():
                print("Preparing data numpys for fold {} vid {}".format(fold, vid))
                vidi_clips = clips_df[clips_df['video fn'].isin([vid])]
                full_vid = np.load(r"/datashare/APAS/features/fold" + str(fold)+r"/" +vid +".npy")
                full_sensor = np.load(r"/datashare/APAS/kinematics_npy/" +vid +".npy")
                for rw in range(len(vidi_clips)):
                    #store clip_i, clip_length, fts
                    try:
                        allvid_clips_data[clipi,:,:] = np.transpose(full_vid[:, vidi_clips["first"].iloc[rw]:vidi_clips["last"].iloc[rw]+1])
                        allsens_clips_data[clipi,:,:] = np.transpose(full_sensor[:, vidi_clips["first"].iloc[rw]:vidi_clips["last"].iloc[rw]+1])
                        alllab_clips_data[clipi,:] = onehot_dict[vidi_clips['gesture'].iloc[rw]]
                        clipi += 1
                    except:
                        print('vid ', vid)
                        print('first ', vidi_clips["first"].iloc[rw])
                        print('last ', vidi_clips["last"].iloc[rw])
                        print('vidfts len ', full_vid.shape)
                        print('sensfts len ', full_sensor.shape) # need to delelte a line from clips_30_100 fro P024_ballon1 since it contained a frame out of range
                        assert(0)
            
            # Transform sensor here
            norm_consts = pd.read_csv("sensor_velocity_statistics_fold_{}.csv".format(fold))
            means = np.expand_dims(np.array(norm_consts.iloc[1][1:]),axis=0)
            stds  = np.expand_dims(np.array(norm_consts.iloc[2][1:]),axis=0)
            allsens_clips_data[allsens_clips_data> 180] -= 360
            allsens_clips_data[allsens_clips_data<-180] += 360
            # broadcast the normalization clip.shape = (36,timesteps) means.shape =(36,1)      
            allsens_clips_data = (allsens_clips_data-means)/stds 
            
            with open(r"/home/student/Desktop/fproj/ds_npys/"+vidftspath,"wb") as f:  
                pickle.dump(allvid_clips_data, f) 
            with open(r"/home/student/Desktop/fproj/ds_npys/"+sensorftspath,"wb") as f:  
                pickle.dump(allsens_clips_data, f) 
            with open(r"/home/student/Desktop/fproj/ds_npys/"+labelpath,"wb") as f:  
                pickle.dump(alllab_clips_data, f) 
            
        
    #with open("ds_npys/"+labelpath, 'rb') as f:
    #    # The protocol version used is detected automatically, so we do not
    #    # have to specify it.
    #    data = pickle.load(f)
    
    
    
    
    
    
    
    
    
    
    
