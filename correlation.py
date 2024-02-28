import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep
mplhep.set_style("CMS")
import numpy as np
import sys
import os

def plotMass(df, name, color="blue", histtype="step"):
  #plt.hist(df.Diphoton_mass, range=(65, 120), bins=50, weights=df.weight)
  plot(df, "Diphoton_mass", name, range_=(65, 120), bins=25, color=color, histtype=histtype)

  plt.title(name)
  plt.xlabel(r"$m_{\gamma\gamma}$")
  plt.ylabel("Count")
  plt.savefig("CorrelationPlots/mass_%s.png"%name)
  plt.clf()

def plot(df, column, label, color="blue", weight_col="weight", range_=(0, 1), bins=5, histtype="step"):
  MC, edges = np.histogram(df[column], range=range_, bins=bins, weights=df[weight_col])
  MC2, edges = np.histogram(df[column], edges, weights=df[weight_col]**2)

  plt.hist(edges[:-1], edges, histtype=histtype, color=color, label=label, weights=MC)
  plt.fill_between(edges, np.append(MC-np.sqrt(MC2), 0), np.append(MC+np.sqrt(MC2), 0), step="post", alpha=0.25, color=color, zorder=10) #background uncertainty

  return MC, MC2

def get_MX_MY(sig_proc):
  if "NMSSM" in sig_proc:
    split_name = sig_proc.split("_")
    MX = float(split_name[10])
    MY = float(split_name[12])
  else:
    raise Exception("Unexpected signal process: %s"%sig_proc)
  return MX, MY

def get_x_label(sig_proc):
  MX, MY = get_MX_MY(sig_proc)
  return r"Output score: $m_X=%d, m_Y=%d$ [GeV]"%(MX, MY)

os.makedirs("CorrelationPlots", exist_ok=True)

veto_df = pd.read_parquet(sys.argv[1])
inv_veto_df = pd.read_parquet(sys.argv[2])

print("\n".join(veto_df))
print()
print("\n".join(inv_veto_df))

plt.hist(veto_df[veto_df.classification==1].weight, range=(-5, 10), bins=30)
plt.savefig("CorrelationPlots/weights.png")
plt.yscale("log")
plt.savefig("CorrelationPlots/weights_log.png")
plt.clf()

print(veto_df.weight.sum())

plotMass(veto_df, "Veto")
plotMass(inv_veto_df, "Inverted Veto")

plotMass(veto_df[veto_df.classification==1], "Veto Peaking")
plotMass(inv_veto_df[inv_veto_df.classification==1], "Inverted Veto Peaking")

plotMass(veto_df[veto_df.classification!=1], "Veto Non-peaking")
plotMass(inv_veto_df[inv_veto_df.classification!=1], "Inverted Veto Non-peaking")

# plotMass(veto_df[veto_df.classification==0], "Unpeaking veto")
# plotMass(inv_veto_df[inv_veto_df.classification==0], "Unpeaking inverted veto")

# plotMass(veto_df[veto_df.classification==3], "Zmmg veto")
# plotMass(inv_veto_df[inv_veto_df.classification==3], "Zmmg inverted veto")

veto_df_peak = veto_df[veto_df.classification == 1]
inv_veto_df_peak = inv_veto_df[inv_veto_df.classification == 1]

veto_df_non = veto_df[veto_df.classification != 1]
inv_veto_df_non = inv_veto_df[inv_veto_df.classification != 1]

print(veto_df_peak.weight.sum(), veto_df_non.weight.sum())

veto_df.info()
inv_veto_df.info()

#veto_df.loc[:, "weight"] *= (1 / veto_df.weight.sum())
inv_veto_df_peak.loc[:, "weight"] *= 1 / inv_veto_df_peak.weight.sum()
veto_df_peak.loc[:, "weight"] *= 1 / veto_df_peak.weight.sum()

for score in filter(lambda x: ("intermediate" in x) and ("MY_90" in x), veto_df_peak.columns):
  v = plot(veto_df_peak, score, "Veto Peaking", "blue")
  inv = plot(inv_veto_df_peak, score, "Inverted Veto Peaking", "red")

  chi2 = np.sum((v[0]-inv[0])**2 / (v[1]+inv[1]))
  plt.title(r"$\chi^2 / d.o.f = %.2f$"%(chi2/5))

  plt.xlabel(get_x_label(score))
  plt.ylabel("Count")
  plt.legend()
  plt.ylim(0, 0.35)
  plt.savefig("CorrelationPlots/%s.png"%score)
  plt.clf()

inv_veto_df_non.loc[:, "weight"] *= 1 / inv_veto_df_non.weight.sum()
veto_df_non.loc[:, "weight"] *= 1 / veto_df_non.weight.sum()

for score in filter(lambda x: ("intermediate" in x) and ("MY_90" in x), veto_df_peak.columns):
  v = plot(veto_df_non, score, "Veto Non-peaking", "blue")
  inv = plot(inv_veto_df_non, score, "Inverted Veto Non-peaking", "red")

  chi2 = np.sum((v[0]-inv[0])**2 / (v[1]+inv[1]))
  plt.title(r"$\chi^2 / d.o.f = %.2f$"%(chi2/5))

  plt.xlabel(get_x_label(score))
  plt.ylabel("Count")
  plt.legend()
  plt.ylim(0, 0.35)
  plt.savefig("CorrelationPlots/non_%s.png"%score)
  plt.clf()