import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sys
import scipy.stats as sps
import scipy.optimize as spo
import scipy.integrate as spi
import mplhep as hep
hep.set_style("CMS")

def dcb(x, mean, sigma, beta1, m1, beta2, m2):
  beta1, m1, beta2, m2 = np.abs(beta1), np.abs(m1), np.abs(beta2), np.abs(m2)

  with np.errstate(all='ignore'):
    A1 = np.power(m1/beta1, m1) * np.exp(-beta1**2/2)
    B1 = m1/beta1 - beta1
    A2 = np.power(m2/beta2, m2) * np.exp(-beta2**2/2)
    B2 = m2/beta2 - beta2

    xs = (x-mean)/sigma

    middle = np.exp(-xs**2/2)*(-beta1 < xs)*(xs < beta2)
    left = np.nan_to_num(A1*np.power(B1-xs, -m1)*(xs<=-beta1), nan=0.0)
    right = np.nan_to_num(A2*np.power(B2+xs, -m2)*(xs>=beta2), nan=0.0)

  return left + middle + right

def f(x, N, mu, sigma, beta1, m1, beta2, m2, a, frac):
  #cb_norm = sps.crystalball.cdf(150, beta, m, loc=mu, scale=sigma) - sps.crystalball.cdf(70, beta, m, loc=mu, scale=sigma)
  #cb = sps.crystalball.pdf(x, beta, m, loc=mu, scale=sigma) / cb_norm

  cb_f = lambda x: dcb(x, mu, sigma, beta1, m1, beta2, m2)

  cb_norm = sum(cb_f(np.arange(70, 150)))
  #cb_norm = spi.quad(cb_f, 70, 150, epsrel=0.01)[0]
  cb = cb_f(x) / cb_norm

  exp_norm = (100/a) * (np.exp(-a*70/100) - np.exp(-a*150/100))
  exp = np.exp(-a*x/100) / exp_norm
  return N * (frac*cb + (1-frac)*exp)

def histogram(df):
  range_ = (70, 150)
  bins = 34
  column = "Diphoton_mass"
  weight_col = "weight"
  MC, edges = np.histogram(df[column], range=range_, bins=bins, weights=df[weight_col])
  MC2, edges = np.histogram(df[column], edges, weights=df[weight_col]**2)
  bin_centers = (edges[:-1] + edges[1:]) / 2
  uncert = np.sqrt(MC2)
  uncert[uncert==0] = min(uncert[uncert!=0])

  bin_width = bin_centers[1] - bin_centers[0]

  factor = sum(MC)*bin_width
  MC /= factor
  uncert /= factor

  return MC, uncert, bin_centers

def plot(MC, uncert, bin_centers, f, popt, savepath, ratio_ylim, bonly=False):
  fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [3, 1]})

  bkg_popt = popt.copy()
  bkg_popt[0] *= (1-bkg_popt[-1])
  bkg_popt[-1] = 0

  x = np.linspace(70, 150, 500)

  axs[0].errorbar(bin_centers, MC, yerr=uncert, fmt="ko")
  if not bonly:
    axs[0].plot(x, f(x, *popt), label="DCB + Exp fit")
  axs[0].plot(x, f(x, *bkg_popt), "--", label="Exp component", zorder=0)

  axs[0].legend()
  axs[0].set_ylabel("Events / (1 GeV)")

  if not bonly:
    ratio = MC / f(bin_centers, *popt)
    ratio_uncert = uncert / f(bin_centers, *popt)
  else:
    ratio = MC / f(bin_centers, *bkg_popt)
    ratio_uncert = uncert / f(bin_centers, *bkg_popt)

  axs[1].errorbar(bin_centers, ratio, yerr=ratio_uncert, fmt="ko")
  axs[1].set_ylim(ratio_ylim)
  axs[1].axhline(1, color="k", linestyle="dashed")

  axs[1].set_xlabel(r"$m_{\gamma\gamma}$")
  axs[1].set_ylabel("MC / Fit")

  plt.savefig(savepath)

df = pd.read_parquet(sys.argv[1])
df_inv = pd.read_parquet(sys.argv[2])

# s+b fit to inverted

MC, uncert, bin_centers = histogram(df_inv)

p0 = [1, 90, 3, 1, 2, 1, 2, 4, 0.95]
bounds = [[0.5, 2],
          [85, 95],
          [2, 4],
          [0.5, 5],
          [1.01, 10],
          [0.5, 5],
          [1.01, 10],
          [1, 10],
          [0.01, 0.99]]
bounds = np.array(bounds).T
res = spo.curve_fit(f, bin_centers, MC, p0, uncert, bounds=bounds)
popt = res[0]

print(f"Inv falling bkg: {df_inv.weight.sum()*(1-popt[0])}")

plot(MC, uncert, bin_centers, f, popt, "inverted_fit.png", (0.5, 1.5))

sig_popt = popt.copy()
sig_popt[0] *= sig_popt[-1]
sig_popt[-1] = 1

MC_minus_DCB = MC - f(bin_centers, *sig_popt)
plot(MC_minus_DCB, uncert, bin_centers, f, popt, "inverted_fit_minus DCB.png", (0.5, 1.5), bonly=True)

# s+b fit to veto with s model from inverted

MC, uncert, bin_centers = histogram(df)

p0 = popt.copy()
p0[-1] = 0.5

bounds[0,1:-1] = popt[1:-1] - 0.001
bounds[1,1:-1] = popt[1:-1] + 0.001

print(list(popt))
print(list(bounds[0]))
print(list(bounds[1]))

res = spo.curve_fit(f, bin_centers, MC, p0, uncert, bounds=bounds)
popt = res[0]
print(popt)

print(f"veto falling bkg: {df.weight.sum()*(1-popt[0])}")

plot(MC, uncert, bin_centers, f, popt, "veto_fit.png", (-1, 3))

# non-peaking veto
MC, uncert, bin_centers = histogram(df[df.classification!=1])

p0 = [1, 90, 3, 1, 2, 1, 2, 4, 0.95]
bounds = [[0.5, 2],
          [85, 95],
          [2, 4],
          [0.5, 5],
          [1.01, 10],
          [0.5, 5],
          [1.01, 10],
          [1, 10],
          [0.01, 0.99]]

bounds = np.array(bounds).T
res = spo.curve_fit(f, bin_centers, MC, p0, uncert, bounds=bounds)
popt = res[0]

plot(MC, uncert, bin_centers, f, popt, "veto_non_peaking.png", (0.5, 1.5), bonly=True)