import sys
import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from run_validation import run_val

def incremental_avg(old_n, old_avg, value):
    """
    Helper function computes the new average incrementally
    """
    return old_avg + (value-old_avg) / (old_n+1)


# Where is the raw Data Stored
path_to_data = "/Users/carelchay/Desktop/School/Modules/DSA4262/Project 2/data/data.json"
# Where are the data labels stored
path_to_labels = "/Users/carelchay/Desktop/School/Modules/DSA4262/Project 2/data/data.info"
# Where to place model parameter files for each epoch
path_to_params = "/Users/carelchay/Desktop/School/Modules/DSA4262/Project 2/data/model_params"
# Where to output the validation dataset with attached probability scores
path_to_val_output = "/Users/carelchay/Desktop/School/Modules/DSA4262/Project 2/data/val_output.csv"
# Where to output the dataframe of epoch vs avg loss
path_to_epoch_output = "/Users/carelchay/Desktop/School/Modules/DSA4262/Project 2/data/epoch_output.csv"


# How many entries from the raw data to use
num_entries = 5000
# Batchsize for training
batchsize = 128
# Learning rate
learning_rate = 0.01
# Number of Epochs
num_epochs = 5

print(f"Using {num_entries:,} entries with batchsize : {batchsize}\n")

# Add scripts folder to path
import sys
pathname="/Users/carelchay/Desktop/School/Modules/DSA4262/Project 2/DSA4262-frontasticfour/scripts"
if pathname not in sys.path:
    sys.path.append(pathname)

### Get the training data ###
print("Importing & Obtaining label Dataframe . . . ", end= "")
import getDataCarel as gd
getData = gd.getData(path_to_data=path_to_data, path_to_labels=path_to_labels)
# Get the dataframe with the labels
label_df = getData.get_labels()
print("Done\n")


### Data Loader class ###
print("Importing & preparing dataloader for TRAINING data . . . ", end= "")
import prepData as prepD
# train - test split based on gene_id (train proportion is 70%).
# This is a list of genes whose entries can be used in the training data
train_genes = prepD.splitdata(label_df = label_df, train=0.85)

# Get the data class for the TRAINING data
trainData = prepD.prepData(train_genes=train_genes, train_set=True,
                          path_to_data=path_to_data, path_to_labels=path_to_labels,
                          num_entries=num_entries)

# Get the dataloader object
trainDataLoader = trainData.get_data_loader(batchsize=batchsize)
training_size = len(trainData)

num_steps = math.ceil(training_size/batchsize)

# Get the data class for the EVAL data
evalData = prepD.prepData(train_genes=train_genes, train_set=False,
                          path_to_data=path_to_data, path_to_labels=path_to_labels,
                          num_entries=num_entries)

# Get the dataloader object for the EVAL data
evalDataLoader = evalData.get_data_loader(batchsize=batchsize, oversample=False, shuffle=False)



print("Done")
print(f"Total of {training_size} entries in training set, with batchsize : {batchsize}, total of {num_steps} steps per epoch. \n")

### Creation of model ###
print("Importing & creating instance of neural network . . . ", end= "")
from m6aNet import m6aNet
model = m6aNet(batchsize=batchsize)
print("Done\n")

### Training Parameters ###

# Training Parameters

# Automatically implements the MSE formula
criterion = nn.BCELoss()
# Use Stochastic Gradient Descent. Need to supply the model parameters
# and a selected learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)
print(f"Using {num_epochs} epochs, learning rate = {learning_rate}")

### Training Loop ###

# Store loss at each epoch
losses = np.array([])

# halfway done with one epoch
stepsize= round(0.5*num_steps)
# Store all the avg losses for each epoch
epoch_loss = []
# Store all the validation losses for each epoch
validation_losses = []

# Start with large loss
last_loss = 100
# Can exceed this no. of times
patience = 1
# Track no. of times it exceeds
triggertimes = 0



for epoch in range(num_epochs):
    model.train()
    step=0
    # The average loss for this epoch
    avg_epoch_loss = 0
    for i, (features, labels) in enumerate(trainDataLoader):
        # Flatten labels
        labels = labels.flatten().float()

        # Forward pass and loss calculation
        outputs = model(features)
        loss = criterion(outputs, labels)

        # update the average epoch loss
        avg_epoch_loss = incremental_avg(old_n=step, old_avg=avg_epoch_loss, value=loss.item())
        step+=1

        # Backward & Optimize
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


        if (i + 1) % stepsize == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{num_steps}], Loss: {loss.item():.4f}, Avg Loss this epoch : {avg_epoch_loss:.4f}')

    # Record the avg loss for this epoch
    epoch_loss.append(avg_epoch_loss)

    # Save the model state at this epoch
    epoch_path = f"{path_to_params}/epoch_{epoch}.pth"
    torch.save(model.state_dict(), epoch_path)

    # Compute the validation loss for this epoch
    current_loss = run_val(model, evalDataLoader, criterion)
    print(f'Avg validation loss for epoch {epoch+1} is {current_loss:.4f}\n')
    validation_losses.append(current_loss)

    if current_loss > last_loss:
        triggertimes +=1
        print(f"Trigger times : {triggertimes}")
        last_loss = current_loss
        if triggertimes > patience:
            print(f"Validation loss not improving. Stopping training early at epoch {epoch+1}")
            break
    else:
        last_loss = current_loss





# print(f"\nAvg Epoch Losses : {epoch_loss}\n")

# Output the epoch num vs loss dataframe
tdf = pd.DataFrame({
    "epoch_num" : [i for i in range(epoch+1)],
    "avg_training_loss" : epoch_loss,
    "avg_validation_loss" : validation_losses
})

tdf.to_csv(path_to_epoch_output, index=False)

#### Perform Evaluation ####

# Instantiate model again
model = m6aNet(batchsize=batchsize)
# Loads the last stored model parameters
model.load_state_dict(torch.load(epoch_path))
model.eval()

# # Get the data class for the EVAL data
# data = prepD.prepData(train_genes=train_genes, train_set=False,
#                           path_to_data=path_to_data, path_to_labels=path_to_labels,
#                           num_entries=num_entries)
#
# # Get the dataloader object for the EVAL data
# dataLoader = data.get_data_loader(batchsize=batchsize, oversample=False, shuffle=False)

# Probability Threshold for M6A modification
threshold = 0.8

# Store prediction scores
results_scores = torch.tensor([])
# Store Prediction Classes
results_class = torch.tensor([])

# Test the model: we don't need to compute gradients
with torch.no_grad():
    n_correct = 0
    n_samples = 0

    for features, labels in evalDataLoader:
        # Flatten labels
        labels = labels.flatten().float()

        # Obtain prediction probabilities
        y_predict = model(features)
        # Store Predicted probabilities
        results_scores = torch.concat([results_scores, y_predict])
        # Convert to classification based on threshold
        y_predict = (y_predict > threshold).float()
        # Store Predicted classes
        results_class = torch.concat([results_class, y_predict])

        # No. of correct predictions
        n_correct += (y_predict == labels).sum().item()
        n_samples += len(labels)

    acc = n_correct / n_samples
    print(f'Accuracy of the network on the {n_samples} eval samples: {100 * acc} %')

# Attach to val_output dataframe
val_output = evalData.df[['gene_id', 'transcript', 'position', 'label']].copy()
val_output['pred_score'] = np.array(results_scores)

# output to path
val_output.to_csv(path_to_val_output, index=False)
