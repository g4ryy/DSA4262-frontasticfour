import json
import numpy as np
import pandas as pd
from pathlib import Path
import os


"""
This python file takes a raw data json file and converts it to a dataframe of columns : 

[transcript, position, k-mer bases, f1_stats , . . . . f9_stats] where each of the fi_stats is a list of 

6 elements denoting the [mean, min, 1st quartile, median, 3rd quartile, max] for that reading
"""

file_path = os.path.join(Path(os.getcwd()).parent.parent.absolute(), 'data')

# Where the json file is
fname = os.path.join(file_path, "data.json")
# Where to place the output file
resname = os.path.join(file_path, "parsed_data_carel.pkl")

r = {
    "transcript" : [],
    "position" : [],
    "k-mer bases": [],
    # Comment out since we don't need all the raw values
    # "values" : []
}
tmp = {f"f{i+1}_stats" : [] for i in range(9)}
r.update(tmp)


with open(fname, 'r') as f:
    for idx, line in enumerate(f):
        line = json.loads(line)
        for transcript, sub1 in line.items():
            r['transcript'].append(transcript)
            for position, sub2 in sub1.items():
                r['position'].append(int(position))
                for bases, values in sub2.items():
                    r['k-mer bases'].append(bases)
                    values = np.array(values)
                    #r['values'].append(values)

                    # Computing summary statistics
                    mean = np.mean(values, axis=0).reshape(1, -1)
                    quantiles = np.quantile(values, [0.25, 0.5, 0.75], axis=0)
                    mini = np.min(values, axis=0).reshape(1, -1)
                    maxi = np.max(values, axis=0).reshape(1, -1)
                    combined = np.concatenate((mean, mini, quantiles, maxi), axis=0)

                    for i in range(9):
                        tmp_i = list(combined[:, i].reshape(-1, ))
                        r[f"f{i + 1}_stats"].append(tmp_i)


r = pd.DataFrame(r)

r.to_pickle(resname)
