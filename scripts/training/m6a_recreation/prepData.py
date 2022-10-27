import numpy as np
import pandas as pd
import getData as gd
import torch
from torch.utils.data import WeightedRandomSampler, DataLoader
import pickle

# Where the k-mer dictionary pickle object is stored. Should have 66 unique keys each mapped to a unique index value
path_to_kmer_dict = "/Users/carelchay/Desktop/School/Modules/DSA4262/Project 2/data/k_mer_dict.pkl"



def splitdata(train=0.85, test=0.3, label_df = None):
    """
    Splits train and eval data based on geneID
    Returns
    -------
    list
        list of geneID's to use for training
    """
    if label_df is None:
        getDat = gd.getData()
        # get label_df
        labels = getDat.get_labels()
    else:
        labels = label_df

    threshold = round(train*labels.shape[0])
    gene_counts = labels.groupby("gene_id")['label'].count().reset_index()
    total, dct = 0,  []
    for idx, row in gene_counts.iterrows():
        total += row['label']
        dct.append(row['gene_id'])
        if total >= threshold:
            break
    return dct

class prepData():
    def __init__(self, train_genes=None, train_set=True, path_to_data = None, path_to_labels = None,
                 k_mer_dict_path=path_to_kmer_dict, num_entries=5000):

        if path_to_data is None or path_to_labels is None:
            self.getDat = gd.getData()
        else:
            self.getDat = gd.getData(path_to_data=path_to_data, path_to_labels=path_to_labels)
        # Label Dataframe
        self.labels = self.getDat.get_labels()

        # K-mer dictionary
        with open(k_mer_dict_path, "rb") as f:
            self.k_mers = pickle.load(f)


        # Data df
        self.df = self.getDat.get_data(num_entries=num_entries)

        # Merge Entries
        self.df = pd.merge(self.df, self.labels, how="left", left_on=("transcript", "position"),
                      right_on=("transcript_id", "transcript_position"))

        # Keep key columns
        self.df = self.df[['gene_id', 'transcript', 'position', "k-mer bases", "values", "label"]]

        # If doing the train - eval split (used in conjunction with a splitdata() result)
        if train_genes is not None:
            # Obtain training set
            if train_set:
                self.df = self.df.loc[self.df['gene_id'].isin(train_genes), :]
            # Obtain evaluation set
            else:
                self.df = self.df.loc[~self.df['gene_id'].isin(train_genes), :]

    def __getitem__(self, index):
        # Obtain the row
        x = self.df.iloc[index]
        # Get that read's values
        x_values = x['values']
        # Get that read's k-mers
        x_bases = x['k-mer bases']
        k_mers = self.k_mers
        # Obtain each 5-mer's index
        base_idx = np.array([k_mers[idx] for idx in [x_bases[i:i + 5] for i in range(3)]])
        # Sample 20 random rows without replacement from the data
        x_values = x_values[np.random.choice(x_values.shape[0], 20, replace=False), :]
        # Stitch the values and the k-mer indexes together
        x = np.zeros((20, 12), float)
        x[:, :x_values.shape[1]] = x_values
        x[:, x_values.shape[1]:] = base_idx
        x = torch.from_numpy(x)

        y = np.array([int(self.df.iloc[index, -1])])
        y = torch.from_numpy(y)
        return x, y

    def __len__(self):
        return self.df.shape[0]

    def get_data_loader(self, batchsize = 50, oversample=True,
                        shuffle=False):
        """
        Returns the DataLoader object
        Parameters
        ----------
        oversample

        Returns
        -------

        """
        if oversample:
            # sample both classes (0, 1) equally
            arr = np.array(self.df['label'])
            sizes = [len(arr[arr==0]), len(arr[arr==1])]
            max_class, min_class = np.argmax(sizes), np.argmin(sizes)
            class_weights = [np.max([int(np.floor(sizes[max_class]/i))-2, 1]) for i in sizes]
            # print(class_weights)

            sample_weights = list(map(lambda x : class_weights[x], arr))

            sampler = WeightedRandomSampler(sample_weights,
                                            num_samples = len(sample_weights),
                                            replacement = True)
            loader = DataLoader(self, batch_size = batchsize, sampler = sampler)

        else:
            loader = DataLoader(self, batch_size=batchsize,shuffle=shuffle)

        return loader


class prepInferData():
    """
    Class for producing dataloader for running INFERENCE
    """

    def __init__(self, path_to_data = None, k_mer_dict_path=path_to_kmer_dict, num_entries=5000):

        if path_to_data is None:
            raise TypeError("Must specify path to inference data")
        else:
            self.getDat = gd.getData(path_to_data=path_to_data)


        # K-mer dictionary
        with open(k_mer_dict_path, "rb") as f:
            self.k_mers = pickle.load(f)


        # Data df
        self.df = self.getDat.get_data(num_entries=num_entries)

    def __getitem__(self, index):
        """
        Returns the input vector feature and the position in place of the label (used for tracking later on)
        Parameters
        ----------
        index

        Returns
        -------

        """
        # Obtain the position
        y = self.df.iloc[index, 1]
        y = torch.tensor([y])

        # Obtain the row
        x = self.df.iloc[index]
        # Get that read's values
        x_values = x['values']
        # Get that read's k-mers
        x_bases = x['k-mer bases']
        k_mers = self.k_mers
        # Obtain each 5-mer's index
        base_idx = np.array([k_mers[idx] for idx in [x_bases[i:i + 5] for i in range(3)]])
        # Sample 20 random rows without replacement from the data
        x_values = x_values[np.random.choice(x_values.shape[0], 20, replace=False), :]
        # Stitch the values and the k-mer indexes together
        x = np.zeros((20, 12), float)
        x[:, :x_values.shape[1]] = x_values
        x[:, x_values.shape[1]:] = base_idx
        x = torch.from_numpy(x)

        return x, y

    def __len__(self):
        return self.df.shape[0]

    def get_data_loader(self, batchsize = 50):
        """
        Returns the DataLoader object
        Parameters
        ----------
        oversample

        Returns
        -------

        """
        loader = DataLoader(self, batch_size=batchsize,shuffle=False)

        return loader
