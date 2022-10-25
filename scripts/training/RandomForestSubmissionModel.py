print("Importing Packages . . .", end = "")
import pandas as pd
import sys
import re
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.dummy import DummyClassifier
from sklearn import metrics
from sklearn.metrics import average_precision_score
from sklearn.utils import resample
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
import os
from pathlib import Path

# Add the scripts folder to the path so we can import the helper class later on
pathname="../utils/"
if pathname not in sys.path:
    sys.path.append(pathname)

import getData as gd
from split import train_test_split
from training_pipeline import run_pipeline
print("Done\n")



data_path = os.path.join(Path(os.getcwd()).parent.parent.absolute(), 'data')
reports_path = os.path.join(Path(os.getcwd()).parent.parent.absolute(), 'reports')



# Path to pickle file with all the processed features
path_to_pkl = os.path.join(data_path, "parsed_data_carel.pkl")
# Path to the report output folder
path_to_report = os.path.join(reports_path, "randomForest_trial.txt")
# Path to submission dataset1 (preprocessed pkl file)
path_to_dataset1 = os.path.join(data_path, "submission_datasets", "dataset1_parsed.pkl")
# Path to dataset 1 submission CSV
dataset1_out = os.path.join(reports_path, "dataset1_scoresrf.csv")
# Path to submission dataset2 (preprocessed pkl file)
path_to_dataset2 = os.path.join(data_path, "submission_datasets", "dataset2_parsed.pkl")
# Path to dataset 2 submission CSV
dataset2_out = os.path.join(reports_path, "dataset2_scoresrf.csv")

dataset0_out = os.path.join(reports_path, "dataset0_scoresrf.csv")

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

print("Spliting Data . . .", end="")
X_train, X_test, y_train, y_test = train_test_split(df, 0.8)
print("Done\n")

param_grid = {
    'bootstrap': [True],
    'max_depth': [40],
    'max_features': ['auto'],
    'min_samples_leaf': [1],
    'min_samples_split':[2],
    'n_estimators': [800]
}

print("Begin training . . .")
model = RandomForestClassifier()
rf = run_pipeline(model, param_grid, X_train, y_train)
joblib.dump(rf, 'finalized_model_rf.pkl')
print("Done training . . .")

# Make Prediction
y_pred = rf.predict(X_test)

# Obtain prediction scores
pred = rf.predict_proba(X_test)
# Keep score for "1" class
pred = pred[:, 1]

# Store Confusion matrix
random_forest_report = classification_report(y_test, y_pred)

# df = pd.DataFrame({'True_labels':y_test, 'scores':pred})
# df.to_csv(dataset0_out, index=False)

# Predict everything as the majority class
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, y_train)
y_pred_dummy=dummy_clf.predict(X_test)
dummypred = dummy_clf.predict_proba(X_test)
dummypred = dummypred[:, 1]
# Get AUC ROC score
fpr, tpr, thresholds = metrics.roc_curve(y_test, pred, pos_label=1)
auc_scorerf = metrics.auc(fpr, tpr)

fpr, tpr, thresholds = metrics.roc_curve(y_test, dummypred, pos_label=1)
auc_scoredummy = metrics.auc(fpr, tpr)

# Get PR Score
pr_scorerf = average_precision_score(y_test, pred, pos_label=1)
pr_scoredummy = average_precision_score(y_test, dummypred, pos_label=1)
# Store Confusion Matrix
dummy_report = classification_report(y_test, y_pred_dummy)
print("Done\n")



# Write out the report
print("Writing to report . . .", end="")
with open(path_to_report, "w") as f:
    f.write(f"Random Forest Classification summary report : \n\n{random_forest_report}\n\n")
    f.write(f"Random Forest AUC Score : \n\n{auc_scorerf}\n\n")
    f.write(f"Random Forest PR Score : \n\n{pr_scorerf}\n\n")
    f.write(f"Dummy Classifier (Majority class) summary report : \n\n{dummy_report}\n\n")
    f.write(f"Random Forest AUC Score : \n\n{auc_scoredummy}\n\n")
    f.write(f"Random Forest AUC Score : \n\n{pr_scoredummy}\n\n")
print("Done\n")



### Perform Predictions Dataset 0

df = pd.read_pickle(path_to_pkl)

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
pred = rf.predict_proba(X)
# Keep score for "1" class
pred = pred[:, 1]
# Append prediction scores
df = df[idx_cols].copy()
df['scores'] = pred

# Cleanup columns
df = df.drop("k-mer bases", axis = 1)
df.columns = ["transcript_id", "transcript_position", "score"]
df.to_csv(dataset0_out, index=False)

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
pred = rf.predict_proba(X)
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
pred = rf.predict_proba(X)
# Keep score for "1" class
pred = pred[:, 1]
# Append prediction scores
df = df[idx_cols].copy()
df['scores'] = pred

# Cleanup columns
df = df.drop("k-mer bases", axis = 1)
df.columns = ["transcript_id", "transcript_position", "score"]
df.to_csv(dataset2_out, index=False)
