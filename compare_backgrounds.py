import pandas as pd
import sys
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def plot(df, column, label, color, weight_col="weight", range_=(0, 1)):
  MC, edges = np.histogram(df[column], range=range_, bins=25, weights=df[weight_col])
  MC2, edges = np.histogram(df[column], edges, weights=df[weight_col]**2)

  plt.hist(edges[:-1], edges, histtype="step", color=color, label=label, weights=MC)
  plt.fill_between(edges, np.append(MC-np.sqrt(MC2), 0), np.append(MC+np.sqrt(MC2), 0), step="post", alpha=0.25, color=color, zorder=8) #background uncertainty

  return MC, MC2

df_gen = pd.read_parquet(sys.argv[1])
df = pd.read_parquet(sys.argv[2],columns=["process_id", "Diphoton_mass", "weight_central"])
with open(sys.argv[3], "r") as f: 
  proc_dict = json.load(f)["sample_id_map"]

inv_non = df_gen[df_gen.classification==0]
#inv_non = inv_non[df_gen.intermediate_transformed_score_NMSSM_XYH_Y_gg_H_tautau_MX_300_MY_90>0.5]
diphoton = df[df.process_id.isin([proc_dict["DiPhoton"], proc_dict["DiPhoton_Low"]])]

diphoton.loc[:, "weight_central"] *= (inv_non.weight.sum() / diphoton.weight_central.sum())

#plt.hist(inv_non.Diphoton_mass, range=(65, 200), bins=20, weights=inv_non.weight, label="DY", density=True, histtype="step")
#plt.hist(diphoton.Diphoton_mass, range=(65, 200), bins=20, weights=diphoton.weight_central, label="Diphoton", density=True, histtype="step")
plot(inv_non, "Diphoton_mass", "DY", "blue", range_=(65, 150))
plot(diphoton, "Diphoton_mass", "Diphoton", "red", weight_col="weight_central", range_=(65, 150))

plt.xlabel(r"$m_{\gamma\gamma}$")
plt.legend()
plt.savefig("compare_backgrounds.png")