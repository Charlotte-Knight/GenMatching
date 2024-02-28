import pandas as pd
import sys
import numpy as np
import json

df = pd.read_parquet(sys.argv[1], columns=["process_id", "event", "year"])

df = df[df.process_id == int(sys.argv[2])]

import os
os.makedirs(sys.argv[3], exist_ok=True)
for year in [2016, 2017, 2018]:
  dfy = df[df.year==year]
  with open(os.path.join(sys.argv[3], f"{year}.json"), "w") as f:
    json.dump(list(dfy.event), f)

print(len(np.unique(df.event)))
print(df)

