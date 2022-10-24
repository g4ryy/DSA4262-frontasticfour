import torch
import torch.nn as nn

def run_val(model, valDataLoader, criterion):
    model.eval()
    loss_total = 0
    with torch.no_grad():
        for features, labels in valDataLoader:
            # Flatten labels
            labels = labels.flatten().float()

            # Forward pass and loss calculation
            outputs = model(features)
            loss = criterion(outputs, labels)
            # increment the total loss
            loss_total += loss.item()

    avg_val_loss = loss_total/len(valDataLoader)
    # print(f'Avg validation loss for this epoch is {avg_val_loss:.4f}\n')
    return avg_val_loss
