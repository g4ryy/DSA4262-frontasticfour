import json
import numpy as np
import pandas as pd


class getData :
    def __init__(self,
                 path_to_data="/home/ubuntu/studies/ProjectStorage/data/data.json",
                 path_to_labels="/home/ubuntu/studies/ProjectStorage/data/data.info"):

        ### Raw Data ###
        self.path_to_data = path_to_data
        # Raw Data to return
        self.data = {
            "transcript": [],
            "position": [],
            "k-mer bases": [],
            "values": []
        }

        ### Labels ###
        self.path_to_labels=path_to_labels
        # Labels
        self.label_df = None

    def get_labels(self):
        if not self.label_df :
            with open(self.path_to_labels, "r") as file:
                lines = file.readlines()
            df = list(map(lambda x: x.replace("\n", ""), lines))
            df = list(map(lambda x: x.split(","), df))
            df = pd.DataFrame(df)
            new_header = df.iloc[0]  # grab the first row for the header
            df = df[1:]  # take the data less the header row
            df.columns = new_header  # set the header row as the df header
            self.label_df = df
        return self.label_df

    def get_data(self):
        pass





# # Where the json file is
# fname = "../data/data.json"
# # Where to place the output file
# resname = "../data/parsed_data_carel.pkl"
#
# r = {
#     "transcript" : [],
#     "position" : [],
#     "k-mer bases": [],
#     # Comment out since we don't need all the raw values
#     # "values" : []
# }
# tmp = {f"f{i+1}_stats" : [] for i in range(10)}
# r.update(tmp)
#
#
# with open(fname, 'r') as f:
#     for idx, line in enumerate(f):
#         line=json.loads(line)
#         for transcript, sub1 in line.items():
#             r['transcript'].append(transcript)
#             for position, sub2 in sub1.items():
#                 r['position'].append(int(position))
#                 for bases, values in sub2.items():
#                     r['k-mer bases'].append(bases)
#                     values = np.array(values)
#                     # Comment out since we don't need all the raw values
#                     # r['values'].append(values)
#
#                     # Computing summary statistics
#                     mean = np.mean(values, axis=0).reshape(1, -1)
#                     quantiles = np.quantile(values, [0.25, 0.5, 0.75], axis = 0)
#                     mini = np.min(values, axis=0).reshape(1, -1)
#                     maxi = np.max(values, axis=0).reshape(1, -1)
#                     combined=np.concatenate((mean, mini, quantiles, maxi), axis=0)
#
#                     for i in range(10):
#                         tmp_i=list(combined[:, 0].reshape(-1,))
#                         r[f"f{i+1}_stats"].append(tmp_i)
#
#
# r = pd.DataFrame(r)
#
# r.to_pickle(resname)
