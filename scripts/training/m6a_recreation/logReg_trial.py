print("Importing Packages . . .", end = "")
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.dummy import DummyClassifier
import re
import sys

# Add the scripts folder to the path so we can import the helper class later on
path_to_scripts = ""
if pathname not in sys.path:
    sys.path.append(path_to_scripts)

import getData as gd
print("Done\n")

# Path to pickle file with all the processed features
path_to_pkl="/home/ubuntu/studies/ProjectStorage/data/parsed_data_carel.pkl"
# Path to the report output folder
path_to_report = ""
# Path to submission dataset1 (preprocessed pkl file)
path_to_dataset1 = ""
# Path to dataset 1 submission CSV
dataset1_out = ""
# Path to submission dataset2 (preprocessed pkl file)
path_to_dataset2 = ""
# Path to dataset 2 submission CSV
dataset2_out = ""


##### Data Preparation #####

# Read in pkl file with the consolidated features
print("Reading in pickle data . . .", end = "")
df = pd.read_pickle(path_to_pkl)
df = df[['transcript', 'position',
       'k-mer bases', 'f1_stats', 'f2_stats', 'f3_stats', 'f4_stats',
       'f5_stats', 'f6_stats', 'f7_stats', 'f8_stats', 'f9_stats']]
print("Done\n")
# Get the label dataframe
print("Obtaining Label Data . . .", end="")
getData = gd.getData()
label_df = getData.get_labels()
print("Done\n")

# Merge the two sources
print("Merging Data . . .", end="")
df = pd.merge(df, label_df, how='left', left_on=("transcript", "position"),
             right_on=("transcript_id", "transcript_position"))
print("Done\n")
# Remove unnecessary columns
print("Performing Column Transformations . . .", end="")
df = df.filter(regex=".*_stats|label")

# Each column right now is a column of lists of 6 elements each. Spread them out into separate unique features
feature_cols = list(filter(lambda x : re.search("stats", x) is not None, df.columns))
for idx, col in enumerate(feature_cols):
    df[[f"{idx+1}_{x}" for x in range(1, 7)]] = df.apply(lambda x : x[col],
                                                           result_type="expand",
                                                          axis=1)
# Remove unnecessary columns
df = df[df.columns.difference(feature_cols)]
print("Done\n")
##### Obtain Training Data #####

# data
print("Obtaining Training Data . . .", end="")
X = np.array(df[df.columns.difference(['label'])])
y = np.array(df['label'])

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
print("Done\n")
##### Model Fitting & Evaluation #####

# Logistic regression model with l2 norm regularization (ridge regression)
print("Performing fitting & Evaluation . . .", end="")
LogReg = LogisticRegression(solver="liblinear", penalty="l2")
# Fit model
LogReg.fit(X_train, y_train)
# Make Prediction
y_pred = LogReg.predict(X_test)
# Store Confusion matrix
ridge_LogReg_report = classification_report(y_test, y_pred)

# Predict everything as the majority class
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, y_train)
y_pred_dummy=dummy_clf.predict(X_test)
# Store Confusion Matrix
dummy_report = classification_report(y_test, y_pred_dummy)
print("Done\n")

# Write out the report
print("Writing to report . . .", end="")
with open(path_to_report, "w") as f:
    f.write(f"Logistic Ridge Regression summary report : \n\n{ridge_LogReg_report}\n\n")
    f.write(f"Dummy Classifier (Majority class) summary report : \n\n{dummy_report}")
print("Done\n")

#### Perform predictions on new data ####

## DATASET 1 ##
df = pd.read_pickle(path_to_dataset1)

# Expand the features
feature_cols = list(filter(lambda x : re.search("stats", x) is not None, df.columns))
for idx, col in enumerate(feature_cols):
    df[[f"{idx+1}_{x}" for x in range(1, 7)]] = df.apply(lambda x : x[col],
                                                           result_type="expand",
                                                          axis=1)
df = df[df.columns.difference(feature_cols)]

# Track index columns for final dataframe
idx_cols = ["transcript", "position", "k-mer bases"]

# Obtain feature matrix
X = np.array(df[df.columns.difference(idx_cols)])

# Obtain prediction scores
pred = LogReg.predict_proba(X)
# Keep score for "1" class
pred = pred[:, 1]
# Append prediction scores
df = df[idx_cols].copy()
df['scores'] = pred

# Cleanup columns
df = df.drop("k-mer bases", axis = 1)
df.columns = ["transcript_id", "transcript_position", "score"]
df.to_csv(dataset1_out, index=False)

## DATASET 2 ##
df = pd.read_pickle(path_to_dataset2)

# Expand the features
feature_cols = list(filter(lambda x : re.search("stats", x) is not None, df.columns))
for idx, col in enumerate(feature_cols):
    df[[f"{idx+1}_{x}" for x in range(1, 7)]] = df.apply(lambda x : x[col],
                                                           result_type="expand",
                                                          axis=1)
df = df[df.columns.difference(feature_cols)]

# Track index columns for final dataframe
idx_cols = ["transcript", "position", "k-mer bases"]

# Obtain feature matrix
X = np.array(df[df.columns.difference(idx_cols)])

# Obtain prediction scores
pred = LogReg.predict_proba(X)
# Keep score for "1" class
pred = pred[:, 1]
# Append prediction scores
df = df[idx_cols].copy()
df['scores'] = pred

# Cleanup columns
df = df.drop("k-mer bases", axis = 1)
df.columns = ["transcript_id", "transcript_position", "score"]
df.to_csv(dataset2_out, index=False)

