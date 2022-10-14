import numpy as np
import pandas as pd
import getData as gd
import torch






def splitdata(train=0.7, test=0.3, label_df = None):
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
    def __init__(self, train_genes=None, train_set=True, path_to_data = None, path_to_labels = None):

        if path_to_data is None or path_to_labels is None:
            self.getDat = gd.getData()
        else:
            self.getDat = gd.getData(path_to_data=path_to_data, path_to_labels=path_to_labels)
        # Label Dataframe
        self.labels = self.getDat.get_labels()
        # K-mer dictionary
        self.k_mers = self.getDat.get_unique_kmers()
        # Data df
        self.df = self.getDat.get_data()

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
                self.df = self.df.loc[self.df['gene_id'].isin(train_genes), :]

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







