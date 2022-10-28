import torch
import torch.nn as nn


class m6aNet(nn.Module):
    def __init__(self, batchsize, readsize=20):
        self.batchsize = batchsize
        self.readsize = readsize
        super(m6aNet, self).__init__()
        # Embedding Layer
        self.embed = nn.Embedding(66, 2)

        ## First Layer ##
        self.read_level_prob_1 = nn.Linear(15, 150)
        # First Batch Norm Layer
        self.norm_1 = nn.BatchNorm1d(num_features=150)
        # First Activation Layer
        self.activ_1 = nn.ReLU()
        # First Dropout Layer
        self.drop_1 = nn.Dropout(p=0.00)

        ## Second Layer ##
        self.read_level_prob_2 = nn.Linear(150, 32)
        # Second Activation Layer
        self.activ_2 = nn.ReLU()
        # Second Dropout Layer
        self.drop_2 = nn.Dropout(p=0.00)

        ## Third Layer ##
        self.read_level_prob_3 = nn.Linear(32, 1)
        # Sigmoid Activation
        self.sig_1 = nn.Sigmoid()

    def forward(self, x):
        ### X is a tensor of shape (batchsize, readsize=20, 12) ###

        # Extract numeric features
        numerics = x[:, :, :9]
        # # Extract Bases
        bases = x[:, :, 9:].type(torch.int64)
        # # Feed to embedding layer
        bases = self.embed(bases)

        # Reshape
        bases = bases.reshape(-1, self.readsize, 3 * 2)
        # Combine embedded output with numeric features
        x = torch.concat((numerics, bases), 2).type(torch.float)

        #### Feed Forward  ####

        ## First Layer ##
        x = self.read_level_prob_1(x)
        # First Batch Norm Layer
        x = x.transpose(dim0=1, dim1=2)  # Need to transpose first
        x = self.norm_1(x)
        x = x.transpose(dim0=1, dim1=2)  # Then transpose back
        # First Activation Layer
        x = self.activ_1(x)
        # First Dropout Layer
        x = self.drop_1(x)

        ## Second Layer ##
        x = self.read_level_prob_2(x)
        # Second Activation Layer
        x = self.activ_2(x)
        # Second Dropout Layer
        x = self.drop_2(x)

        ## Third Layer ##
        x = self.read_level_prob_3(x)
        # Sigmoid Activation
        x = self.sig_1(x)
        x = x.reshape(-1, self.readsize)

        # Final Output
        r = 1 - torch.prod(1 - x, axis=1)
        return r