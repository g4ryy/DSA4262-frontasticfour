import sys
import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import warnings
import time
import argparse


parser = argparse.ArgumentParser(description='Run the inference of m6ANet on unlabelled data')
parser.add_argument("ptd", type=str, help="Path to datafile (json format)")
parser.add_argument("ptp", type=str, help="Path to parameters folder. Which model parameters to use \n(Should be a .pth file)")
parser.add_argument("pto", type=str, help="Path to output csv file. Where to output the dataset with attached probability scores")

parser.add_argument("-nentries", type=int, help="How many entries from the raw data to use. \nSet to 0 to use all data.", default=1000)
parser.add_argument("-batchsize", type=int, help="Batchsize to use", default=128)
parser.add_argument("-il", type=int, help="No. of inference loops to perform before averaging the results", default=200)
parser.add_argument("-utilspath", type=str, help="Path to the folder with Utility scripts", default="../../utils/")


args = parser.parse_args()

# Add scripts folder to path
import sys
if args.utilspath not in sys.path:
    sys.path.append(args.utilspath)



# # Where is the raw Data Stored
# path_to_data = "/Users/carelchay/Desktop/School/Modules/DSA4262/Project 2/data/data.json"
# # Which model parameter to use
# path_to_params = "/Users/carelchay/Desktop/School/Modules/DSA4262/Project 2/data/model_params/epoch_4.pth"
# # Where to output the dataset with attached probability scores
# path_to_output = "/Users/carelchay/Desktop/School/Modules/DSA4262/Project 2/data/val_output256.csv"


# How many entries from the raw data to use
# num_entries = 1000
# Batchsize for Inference. Don't change this
# batchsize = 256
# How many loops to perform before averaging the results
# infer_loops = 200

### Data Loader class ###
print("\nImporting & preparing dataloader for INFERENCE data . . . ", end= "")
import prepData as prepD
inferData = prepD.prepInferData(path_to_data=args.ptd, num_entries=args.nentries)
inferDataLoader = inferData.get_data_loader(batchsize=args.batchsize)
print("Done\n")

infersize = len(inferData)


### Creation of model ###
print("Importing & creating instance of neural network . . . ", end= "")
from m6aNet import m6aNet
model = m6aNet(batchsize=args.batchsize)
# Loads the last stored model parameters
model.load_state_dict(torch.load(args.ptp))
model.eval()
print("Done\n")


print(f"Total of {infersize} entries in inference set, with batchsize : {args.batchsize}. Total of {len(inferDataLoader)} steps per loop\n")

### Run Inference ###

# Store all scores for averaging
all_scores = torch.tensor([])
# Store all positions to ensure correct positioning
all_positions = torch.tensor([])

# Test the model: we don't need to compute gradients
start_time = time.time()
with torch.no_grad():
    for runs in range(args.il) :
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
            print(f"Completed run {runs+1}/{args.il}, Time Taken : {mins[0]} Mins, {mins[1]:.2f} seconds\n")
            start_time = time.time()


# Reshape scores and compute the average for each read
all_scores = all_scores.reshape(args.il, -1).mean(dim=0)

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
# Check that directory exists otherwise create it
os.makedirs(os.path.dirname(args.pto), exist_ok=True)
res_data.to_csv(args.pto, index=False)

print(f"\nCompleted score calculations on dataset with {args.il} averaged scores")

