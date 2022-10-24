import sys
import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import warnings
import time



# Where is the raw Data Stored
path_to_data = "/Users/carelchay/Desktop/School/Modules/DSA4262/Project 2/data/data.json"
# Where are the data labels stored
path_to_labels = "/Users/carelchay/Desktop/School/Modules/DSA4262/Project 2/data/data.info"
# Which model parameter to use
path_to_params = "/Users/carelchay/Desktop/School/Modules/DSA4262/Project 2/data/model_params/epoch_4.pth"
# Where to output the dataset with attached probability scores
path_to_output = "/Users/carelchay/Desktop/School/Modules/DSA4262/Project 2/data/val_output256.csv"


# How many entries from the raw data to use
num_entries = 1000
# Batchsize for Inference. Don't change this
batchsize = 256
# How many loops to perform before averaging the results
infer_loops = 200

### Data Loader class ###
print("Importing & preparing dataloader for INFERENCE data . . . ", end= "")
import prepData as prepD
inferData = prepD.prepInferData(path_to_data=path_to_data, num_entries=num_entries)
inferDataLoader = inferData.get_data_loader(batchsize=batchsize)
print("Done\n")

print(f"Total of {len(inferDataLoader)} steps\n")

infersize = len(inferData)


### Creation of model ###
print("Importing & creating instance of neural network . . . ", end= "")
from m6aNet import m6aNet
model = m6aNet(batchsize=batchsize)
# Loads the last stored model parameters
model.load_state_dict(torch.load(path_to_params))
model.eval()
print("Done\n")


print(f"Total of {infersize} entries in inference set, with batchsize : {batchsize}\n")

### Run Inference ###

# Store all scores for averaging
all_scores = torch.tensor([])
# Store all positions to ensure correct positioning
all_positions = torch.tensor([])

# Test the model: we don't need to compute gradients
start_time = time.time()
with torch.no_grad():
    for runs in range(infer_loops) :
        for features, positions in inferDataLoader:
            if runs ==0:
                # Flatten positions
                positions = positions.flatten().float()
                all_positions = torch.concat([all_positions, positions])

            # Obtain prediction probabilities
            y_predict = model(features)
            # Store Predicted probabilities
            all_scores = torch.concat([all_scores, y_predict])

        if (runs+1)%50 ==0:
            total_time = time.time() - start_time
            mins = divmod(total_time, 60)
            print(f"Completed run {runs+1}/{infer_loops}, Time Taken : {mins[0]} Mins, {mins[1]:.2f} seconds\n")
            start_time = time.time()


# Reshape scores and compute the average for each read
all_scores = all_scores.reshape(infer_loops, -1).mean(dim=0)

# Check that the positions match
orig_positions = inferData.df['position'].copy()
non_matching_positions = np.array(all_positions) == orig_positions
non_matching_positions = list(non_matching_positions[~non_matching_positions].index)

if len(non_matching_positions) >0:
    print(f"Not all positions match. Mismatch on indexes : {non_matching_positions}")
else:
    print("All positions match")

res_data = inferData.df[['transcript', 'position']].copy()
res_data['score'] = np.array(all_scores)
# rename columns
res_data.columns = ['transcript_id', 'transcript_position', 'score']
# Write to CSV
res_data.to_csv(path_to_output, index=False)

print(f"\nCompleted score calculations on dataset with {infer_loops} averaged scores")

