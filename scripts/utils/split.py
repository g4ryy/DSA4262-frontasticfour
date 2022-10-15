import numpy as np
import pandas as pd
import re
from sklearn.utils import resample


# function to split the data based on gene_id and balance the labels in training and testing sets
def train_test_split(df, train_proportion=0.8):
    train_size= int(len(df) * train_proportion)
    
    true_labels = df[df['label'] == '1']
    true_genes = true_labels['gene_id'].unique()
    # random.shuffle(true_genes)
    
    train_df = pd.concat([true_labels[true_labels['gene_id'] == i] for i in true_genes[0:int(len(true_genes) / 2)]])
    test_df = pd.concat([true_labels[true_labels['gene_id'] == i] for i in true_genes[(int(len(true_genes) / 2)):]])
    
    false_labels = df[df['label'] == '0']
    for gene in false_labels['gene_id'].unique():
        if gene in train_df['gene_id'].unique():
            train_df = pd.concat([train_df, false_labels[false_labels['gene_id'] == gene]])
        elif gene in test_df['gene_id'].unique():
            test_df = pd.concat([test_df, false_labels[false_labels['gene_id'] == gene]])
        else:
            if len(train_df) < train_size:
                train_df = pd.concat([train_df, false_labels[false_labels['gene_id'] == gene]])
            else:
                test_df = pd.concat([test_df, false_labels[false_labels['gene_id'] == gene]])
                
    train_df = train_df.filter(regex=".*_stats|label")
    test_df = test_df.filter(regex=".*_stats|label")
    feature_cols = list(filter(lambda x : re.search("stats", x) is not None, train_df.columns))
    train_df = train_df.copy() 
    test_df = test_df.copy() 
    for idx, col in enumerate(feature_cols):
        train_df[[f"{idx+1}_{x}" for x in range(1, 7)]] = train_df.apply(lambda x : x[col], 
                                                           result_type="expand", 
                                                          axis=1)
        test_df[[f"{idx+1}_{x}" for x in range(1, 7)]] = test_df.apply(lambda x : x[col], 
                                                           result_type="expand", 
                                                          axis=1)
    
    train_df = train_df[train_df.columns.difference(feature_cols)]
    test_df = test_df[train_df.columns.difference(feature_cols)]
    X_train = np.array(train_df[train_df.columns.difference(['label'])])
    X_test = np.array(test_df[test_df.columns.difference(['label'])])
    y_train = np.array(train_df['label']) 
    y_test = np.array(test_df['label']) 
    return X_train, X_test, y_train, y_test