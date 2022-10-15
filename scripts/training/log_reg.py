import pandas as pd
import sys
import os
import re
import random
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.dummy import DummyClassifier
from sklearn import preprocessing
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV


pathname="../utils/"
if pathname not in sys.path:
    sys.path.append(pathname)
    
    
from split import train_test_split
import getData as gd


data_path = os.path.join(Path(os.getcwd()).parent.parent.absolute(), 'data')
reports_path = os.path.join(Path(os.getcwd()).parent.parent.absolute(), 'reports')

# Path to pickle file with all the processed features
path_to_pkl = os.path.join(data_path, "parsed_data_carel.pkl")
# Path to the report output folder
path_to_report = os.path.join(reports_path, "log_reg.txt")
# Path to submission dataset1 (preprocessed pkl file)
path_to_dataset1 = os.path.join(data_path, "submission_datasets", "dataset1_parsed.pkl")
# Path to dataset 1 submission CSV
dataset1_out = os.path.join(reports_path, "dataset1_scoreslr.csv")
# Path to submission dataset2 (preprocessed pkl file)
path_to_dataset2 = os.path.join(data_path, "submission_datasets", "dataset2_parsed.pkl")
# Path to dataset 2 submission CSV
dataset2_out = os.path.join(reports_path, "dataset2_scoreslr.csv")


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


# Split Data
X_train, X_test, y_train, y_test = train_test_split(df, 0.8)
print("Done splitting\n")


# Scale training data
scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


# Grid Search CV
param_grid = {
    'penalty' : ['l1','l2'], 
    'C'       : np.logspace(-3,3,7),
    'solver'  : ['newton-cg', 'lbfgs', 'liblinear'],
}

log_reg = LogisticRegression()

clf = GridSearchCV(log_reg,                    # model
                   param_grid = param_grid,   # hyperparameters
                   scoring='accuracy',        # metric for scoring
                   cv=5)   
clf.fit(X_train, y_train)

print("Tuned Hyperparameters :", clf.best_params_)
print("Accuracy :",clf.best_score_)

best_grid = clf.best_estimator_
# Make Prediction
y_pred = best_grid.predict(X_test)
# Store Confusion matrix
log_reg_report = classification_report(y_test, y_pred)

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
    f.write(f"Logistic Regression Classification summary report : \n\n{log_reg_report}\n\n")
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
pred = log_reg.predict_proba(X)
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
pred = log_reg.predict_proba(X)
# Keep score for "1" class
pred = pred[:, 1]
# Append prediction scores
df = df[idx_cols].copy()
df['scores'] = pred

# Cleanup columns
df = df.drop("k-mer bases", axis = 1)
df.columns = ["transcript_id", "transcript_position", "score"]
df.to_csv(dataset2_out, index=False)