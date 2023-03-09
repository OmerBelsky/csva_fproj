# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 17:36:06 2023

@author: atver
"""

# Collect normalization constants for the kinematic data after overangle adjustment

import glob
import pandas as pd
import numpy as np
import os


colnames = ['Sensor #1 X', 'Sensor #1 Y', 'Sensor #1 Z',
       'Sensor #1 Euler Angles (Z Y\'X\") Z',
       'Sensor #1 Euler Angles (Z Y\'X\") Y\'',
       'Sensor #1 Euler Angles (Z Y\'X\") X\"',
       'Sensor #2 X', 'Sensor #2 Y', 'Sensor #2 Z',
       'Sensor #2 Euler Angles (Z Y\'X\") Z',
       'Sensor #2 Euler Angles (Z Y\'X\") Y\'',
       'Sensor #2 Euler Angles (Z Y\'X\") X\"',
       'Sensor #3 X', 'Sensor #3 Y', 'Sensor #3 Z',
       'Sensor #3 Euler Angles (Z Y\'X\") Z',
       'Sensor #3 Euler Angles (Z Y\'X\") Y\'',
       'Sensor #3 Euler Angles (Z Y\'X\") X\"',
       'Sensor #4 X', 'Sensor #4 Y', 'Sensor #4 Z',
       'Sensor #4 Euler Angles (Z Y\'X\") Z',
       'Sensor #4 Euler Angles (Z Y\'X\") Y\'',
       'Sensor #4 Euler Angles (Z Y\'X\") X\"', 
       'Sensor #5 X', 'Sensor #5 Y', 'Sensor #5 Z',
       'Sensor #5 Euler Angles (Z Y\'X\") Z',
       'Sensor #5 Euler Angles (Z Y\'X\") Y\'',
       'Sensor #5 Euler Angles (Z Y\'X\") X\"', 
       'Sensor #6 X', 'Sensor #6 Y', 'Sensor #6 Z',
       'Sensor #6 Euler Angles (Z Y\'X\") Z',
       'Sensor #6 Euler Angles (Z Y\'X\") Y\'',
       'Sensor #6 Euler Angles (Z Y\'X\") X\"']





if not "atver" in os.getcwd():
    os.chdir("/datashare/APAS/kinematics_npy")
    
    
 # go over all sensor feature files in each train fold
  
 # compile them into a single np
 # apply the overangle compensation
 # run describe 
 # save describe to 

for fold in range(5):
    print("STARTING FOLD: ", fold)
    folds_path = r"/datashare/APAS/folds"
    test_videos = []
    with open(os.path.join(folds_path,"test "+ str(fold) +".txt")) as f:
        for ln in f: # for each line
            test_videos += [ln.split()[0][:-4]] 
    valid_videos = []
    with open(os.path.join(folds_path,"valid "+ str(fold) +".txt")) as f:
        for ln in f: # for each line
            valid_videos += [ln.split()[0][:-4]]
    all_sensors = np.array([])
    for fn in glob.glob("*.npy"):
        if ("31" in fn) or (fn[:-4] in test_videos) or (fn[:-4] in valid_videos):
            print("skipping ", fn)
            continue
        if all_sensors.shape==(0,):
            all_sensors = np.load(r"/datashare/APAS/kinematics_npy/"+fn)
        else:
            new_data = np.load(r"/datashare/APAS/kinematics_npy/"+fn)
            all_sensors = np.concatenate((all_sensors,new_data),axis=1)
    # apply the overangle compensation to angle velocity rows 
    for ang_set in range(6):
        angles = all_sensors[3+ang_set*6:6+ang_set*6,:]
        angles[angles>  180] -= 360
        angles[angles< -180] += 360
        all_sensors[3+ang_set*6:6+ang_set*6,:] = angles
    
    # convert to pandas
    all_sensors = np.transpose(all_sensors)
    df = pd.DataFrame(all_sensors, columns = colnames)
    # run describe 
    desc = df.describe()
    # save panda of fold
    desc.to_csv("/home/student/Desktop/fproj/sensor_velocity_statistics_fold_"+str(fold)+".csv")









