import pandas as pd

path = "/scratch/network/lg0508/SmartPixels/17180303/dataset_2s/dataset_2s_50x10/labels_d16401.parquet"
df = pd.read_parquet(path)

print("Rows:", len(df))
print("Columns:", df.columns.tolist())
print("\nDtypes:\n", df.dtypes)

print("\nHead:\n", df.head(5))

cols = [c for c in ["pt","cotAlpha","cotBeta","n_x","n_y","n_z"] if c in df.columns]
if cols:
    print("\nDescribe (key cols):\n", df[cols].describe())
