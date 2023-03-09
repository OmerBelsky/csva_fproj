# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 14:25:11 2023

@author: atver
"""
import glob
import pandas as pd
import numpy as np
import os

if not "atver" in os.getcwd():
    os.chdir("/datashare/APAS/kinematics")
# go over all files
# get the description statistics for the position xyz of each sensor
# Announce where the minimum is less than -1
# Announce where the max in more than 1
# Announce where the standard deviation is out of range [.2,1.2]

# Apply proposed fixes to the Euler angle and angular velociities and see if they sitll have outllying values ie not in [-1,1]

for vid in glob.glob("*.csv"): # For each vid 
    # position xyz panda
    sensor_data_pd = pd.read_csv(vid)
    d = sensor_data_pd[[("Sensor #{} {}".format(nsensor, ax)) for nsensor in range(1,7) for ax in ["X", "Y","Z"]]].describe()
    
    # announce mins
    minval = -1 # min value will accept
    for col, val in zip(d.loc['min'][d.loc['min'] < minval].index.tolist(), d.loc['min'][d.loc['min'] < minval]):
        print("vid: ",vid, +" col ", col, "minval ", val)
    
    # announce maxs    
    maxval = 1 # max value will accept
    for col, val in zip(d.loc['max'][d.loc['max'] > maxval].index.tolist(), d.loc['max'][d.loc['max'] > maxval]):
        print("vid: ",vid, +" col ", col, "maxval ", val)
        
    # announce stds not between .2 and 1.2
    for col, val in zip(d.loc['std'][d.loc['std'].between(.2, 1.2, inclusive = 'neither')].index.tolist(), d.loc['std'][d.loc['std'].between(.2, 1.2, inclusive = 'neither')]):
        print("vid: ",vid, +" col ", col, "stdval ", val)
    
    # angle panda
    ea = sensor_data_pd[[col for  col in sensor_data_pd.columns if "Euler" in col]]
    ea_mod = ea.copy()
    ea_mod[ea_mod<-180] = ea_mod[ea_mod<-180]+360
    ea_mod[ea_mod> 180] = ea_mod[ea_mod> 180]-360
    ea_modd = ea_mod.describe()
    # announce mins after modifying the angles
    minval = -180 # min value will accept
    for col, val in zip(ea_modd.loc['min'][ea_modd.loc['min'] < minval].index.tolist(), ea_modd.loc['min'][ea_modd.loc['min'] < minval]):
        print("vid: ",vid, +" col ", col, "minval ", val)
    
    # announce maxs  after modifying the angles  
    maxval = 180 # max value will accept
    for col, val in zip(ea_modd.loc['max'][ea_modd.loc['max'] > maxval].index.tolist(), ea_modd.loc['max'][ea_modd.loc['max'] > maxval]):
        print("vid: ",vid, " col ", col, "maxval ", val)
    
    
    # in the numpy sensor ft the rows are the feature 36 for each of 6 sensors xyz velocity and euler angle velocities
    
    
    
    
    



