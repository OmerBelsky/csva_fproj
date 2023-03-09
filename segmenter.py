# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 18:42:54 2023

@author: atver
"""

# Create a list of clips of uniform size and minimum percent activity consistency 
# without overlap

# loop over ALL the videos start tracking where window is in the video start and end
# window will start at the begining of the video
# The last frame in the window will be examined and if it has the same gesture 
# as the majority class by a given threshold of activity consistency then 
# the videofn, startfr, endfr, end class, and percentage consistency are all saved to a pandas df -> csv
# then the window is shifted by the window length
# if the window does not meet this consitency then the window is shifted by a single frame
# when the start of the window is too close to the end of the video - continue
# while nFr - win_s >= win_len:
# This can then be loaded and filtered by the video file names that appear in a fold 

import pandas as pd
import numpy as np
import glob
import os 
import sys
if not "atver" in os.getcwd():
    os.chdir("/datashare/APAS/transcriptions_gestures")
    

def get_gesture_label(vid,frame):
    with open(vid) as f:
        line_number = 1
        for ln in f: # for each line
            first, last, gest = ln.split() # get gest code and its len - add to panda
            if int(first) <= frame <= int(last):
                return gest, line_number
            line_number += 1 
        raise AssertionError("No gesture found for frame {} by the end of file {}".format(frame,vid))

window = int(sys.argv[1])
threshold = float(sys.argv[2])

# window = 60 # the window should encompass this number of frames
# threshold = 0
vidlen_df = pd.DataFrame(columns = ["video fn","gest transcript","k_npy", "k_csv"]) # assume that the gesture files are same length as the features 
vid_len_dict = {col : [] for col in vidlen_df.columns}
df = pd.DataFrame(columns = ["video fn","first","last", "gesture", "consistency"]) 
for vid in glob.glob("*.txt"): # For each vid 
    vid_df = pd.DataFrame(columns = ["video fn","first","last", "gesture", "consistency"])  
    vid_dict = {col : [] for col in vid_df.columns}
    
    with open(vid) as f:
        for ln in f: # for each line
            first,last,gest = ln.split() # get gest code - add to panda
            last_Fr = int(last) # zero indexed
    print(last_Fr+1, " frames in gesture transcript file for ", vid) # should all be integers
    
    knpy_path = os.path.join(r"/datashare/APAS/kinematics_npy", vid[:-4] + '.npy')
    knpy_mmap = np.load(knpy_path, mmap_mode = 'r')
    last_Fr_knpy = knpy_mmap.shape[1] -1 # zero index
    
    kcsv_path = os.path.join(r"/datashare/APAS/kinematics", vid[:-4] + '.csv')
    kcsv_mmap = pd.read_csv(kcsv_path, memory_map = True)
    last_Fr_kcsv = len(kcsv_mmap) -1 # zero index
    
    vid_len_dict["video fn"] += [vid] 
    vid_len_dict["gest transcript"] += [last_Fr] 
    vid_len_dict["k_npy"] += [last_Fr_knpy] 
    vid_len_dict["k_csv"] += [last_Fr_kcsv] 
    
    last_Fr = min(last_Fr, last_Fr_knpy, last_Fr_kcsv)
    win_start = 0 # frame where the window is starting.
    # Assume that all the video and sensor data begin at 0 but may end at 
    # different times. Use the shortest one to ensure we have the same size data!
    while last_Fr+1 - win_start >= window: # there is a full window left in the video
        gest_start, line_start = get_gesture_label(vid, win_start)
        gest_end, line_end = get_gesture_label(vid, win_start+window-1)
        if line_start == line_end:
            consistent = True
            consistency = 1
        else:
            win_gest = [get_gesture_label(vid, x)[0] for x in range(win_start,win_start+window)]
            consistency = win_gest.count(gest_end)/window 
            consistent = consistency >= threshold
            
        if consistent:
            vid_dict["video fn"] = vid[:-4]
            vid_dict["first"] += [win_start]
            vid_dict["last"] += [win_start + window - 1]
            vid_dict["gesture"] += [gest_end]
            vid_dict["consistency"] += [consistency]
            win_start += window
        else:
            win_start += 1
            
    vid_df = pd.DataFrame(vid_dict)
    
    df = pd.concat([df,vid_df])

os.chdir("/home/student/Desktop/fproj")

df.to_csv("clips_{}_{:03d}.csv".format(window,int(threshold*100)))

vidlen_df = pd.DataFrame(vid_len_dict)
vidlen_df.to_csv("vid_file_lengths.csv")

