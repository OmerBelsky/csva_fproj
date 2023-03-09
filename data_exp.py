# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 13:38:17 2023

@author: atver
"""
import numpy as np
gestures = {"No gesture" : "G0",
            "Needle passing" : "G1",
            "Pull the suture" : "G2",
            "Instrument tie" : "G3",
            "Lay the knot" : "G4",
            "Cut the suture" : "G5"}


# vid_fts_f0 =np.load(r"D:\Users\atver\Downloads\CPUVIZfinalproj\some data files\P016_balloon1_f0.npy")
# The vid features come as np.array of shape [1280, nFr] 
# The vid features are different between the different folds. Why?




# Sensor data has 90 columns there are 6 sensors and 15 data points per time step
# The sensor numpy arrays on the server have 36 featues per time point. How were these derived? Are they an embedding?

# Sensor feature numpy arrays and the vid features numpy array are not the same length 
# (difference of 13 time points for p16_ballon1)(difference of 19 time points for p17_ballon1)
# How do we allign the sensor arrays and the videos? 

# Sensor feature numpy arrays and the csvs are they same length.

"P016_balloon1_f0.npy"

# I suspected that the rotation matrix is like the derivative of the euler angles
# Accessed some data to verify this and NO it is another way to show the orientation
# from csv P016_ballon1 row 5
# Euler angels = 178.457327	-25.425327	-174.018593
# rotation matrix = -0.902418	-0.017934	-0.42964	0.024303	0.995398	-0.092632	0.429329	-0.094077	-0.897827
# http://eecs.qmul.ac.uk/~gslabaugh/publications/euler.pdf
# https://towardsdatascience.com/better-rotation-representations-for-accurate-pose-estimation-e890a7e1317f # relative strengths and weakness for rotation representation in ML
rot = np.array([[-0.902418, -0.017934, -0.42964], [0.024303, 0.995398, -0.092632], [0.429329, -0.094077, -0.897827]])
-np.arcsin(rot[2,0])*180/np.pi
np.arctan2(-rot[2,1],rot[2,2])*180/np.pi
np.arctan2(rot[1,0],rot[0,0])*180/np.pi
# The signs need some extra steps
# The take away is that the rot matrx and euler angles encode the same information and are interconvertable 
# There should be no more information in the rot matrix than in the euler angles so let's preferably use them.
# The rotation matrix further more can have ambiguity without the extra steps mentioned that might be harder for the model to learn.

# We can compare training on the rotation matrix and the euler angles alone

# Also show their counts
## Create histograms of the gesture lengths for all classes

# Make a pandas df to hold this info
import pandas as pd
import glob
import os 
if not "atver" in os.getcwd():
    os.chdir("/datashare/APAS/transcriptions_gestures")
    
gests = ["G0","G1","G2","G3","G4","G5"]  
df = pd.DataFrame(columns = gests)  
for vid in glob.glob("*.txt"): # For each vid 
# vid = r"D:\Users\atver\Downloads\CPUVIZfinalproj\some data files\P016_tissue2.txt"
    vid_df = pd.DataFrame(columns = gests)
    vid_dict = {g : [] for g in gests}
    with open(vid) as f:
        for ln in f: # for each line
            first,last,gest = ln.split() # get gest code and its len - add to panda
            for g in gests:
                vid_dict[g] = vid_dict[g]  + [int(last)-int(first)+1] if g == gest else vid_dict[g] + [np.nan]   
    vid_df = pd.DataFrame(vid_dict)
    vid_df["vid"] = vid[:-4]
    df = pd.concat([df,vid_df])

os.chdir("/home/student/Desktop/fproj")
df.to_csv("gesture_lengths")
    
    

# I also want the difference in lengths for the sensor and vid fts
# vid ft - sensor



# display graph per class

# get the lengths of all the videos


 


