import pandas as pd
import sys
import json
from tqdm import tqdm
import numpy as np

def attachClassification(df, ec):
  df["classification"] = -1
  for year in ec.keys():
    for event_id in tqdm(ec[year].keys()):
      #print(sum(df.event==int(event_id)))
      df.loc[df.event==int(event_id), "classification"] = ec[year][event_id]

from pyarrow.parquet import ParquetFile
def getColumns(parquet_file):
  pf = ParquetFile(parquet_file) 
  columns = [each.name for each in pf.schema]
  if "__index_level_0__" in columns:
    columns.remove("__index_level_0__")
  return columns

columns = getColumns(sys.argv[1])
columns = list(filter(lambda x: ("score" not in x) or ("90" in x), columns))
print(columns)

df = pd.read_parquet(sys.argv[1], columns=columns)
df = df[df.process_id.isin([17])]

df.drop_duplicates(subset="event", inplace=True)

uniques, counts = np.unique(df.event, return_counts=True)
assert (counts==1).all()

print(df)
df.sort_values("event", axis=0, inplace=True)
print(df)

with open(sys.argv[2], "r") as f:
  ec = json.load(f)

classification = []
for e in tqdm(df.event):
  classification.append(2)
  for year in ec.keys():
    if str(e) in ec[year].keys():
      classification[-1] = ec[year][str(e)]
      break

print(len(classification), len(df.event))
assert len(classification) == len(df.event)

df["classification"] = classification

#attachClassification(df, ec)
df.to_parquet(sys.argv[3])

#import matplotlib
#matplotlib.use("Agg")
#import matplotlib.pyplot as plt
##peak_df = df[df.classification==1]
#peak_df = df[(df.classification==1)&(df.LeadPhoton_pixelSeed==0)&(df.SubleadPhoton_pixelSeed==0)]
#plt.hist(peak_df.Diphoton_mass, bins=20, range=(50, 130), weights=peak_df.weight_central)
#plt.savefig("peaking_mgg.pdf")
