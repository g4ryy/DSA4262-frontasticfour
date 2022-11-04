import argparse
import os
import pandas as pd

parser = argparse.ArgumentParser(description='Consolidate results for a given folder')
parser.add_argument("folder_path", type=str, help="Path to cell line folder")

args = parser.parse_args()

# Standardize folder name
folder_name = args.folder_path.strip("/")

# Name of cell line
cell_line = os.path.basename(folder_name)


# Get list of results files
r = []
for dirpath, subdir, filenames in os.walk(folder_name):
    results = [fname for fname in filenames if 'results' in fname]
    results = [os.path.join(dirpath, fname) for fname in results]
    r.extend(results)

print(f"Results found in {r}\n")

# Filename
fname = r[0]
# Directory name (replicate name)
dirname = os.path.basename(os.path.dirname(fname))
# Read in csv file
df = pd.read_csv(fname)
# Convert column name score
df.rename(columns={'score': dirname}, inplace=True)

# Join rest of result files
for fname in r[1:]:
    # Directory name (replicate name)
    dirname = os.path.basename(os.path.dirname(fname))
    # Read in csv file
    tdf = pd.read_csv(fname)
    # Convert column name score
    tdf.rename(columns={'score': dirname}, inplace=True)
    # Join with main dataframe
    df = pd.merge(df, tdf, how='outer', on=('transcript_id', 'transcript_position'))

# Columns with scores
score_cols = df.columns.difference(['transcript_id', 'transcript_position'])
score_cols = list(score_cols)

# Additional Columns
df['mean_score'] = df[score_cols].mean(axis=1)
df['sdev'] = df[score_cols].std(axis=1)
df['n_counts'] = df[score_cols].count(axis=1)

print(f"Total of {len(score_cols)} probability score columns\n")

out_path = os.path.join(folder_name, f"{cell_line}_consolidated_results.csv")
df.to_csv(out_path, index=False)
print(f"Consolidated results for cell line {cell_line} written out to {out_path}")