import pandas as pd

from preprocessing import preprocess_df

df = pd.read_pickle(f"riiid_train.pkl.gzip")
print("Preprocessing")
df = preprocess_df(df)

print("done")