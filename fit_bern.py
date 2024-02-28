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

def dcb(x, mean, sigma, beta1, m1, beta2, m2, norm=True):
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

  s = left + middle + right

  if norm:
    return s / sum(dcb(np.arange(65, 150), mean, sigma, beta1, m1, beta2, m2, norm=False))
  else:
    return s

def poly3_nonorm(x, a, b, c):
  x = x / 100
  poly = a*x**3 + b*x**2 + c*x
  return poly

def poly3(x, a, b, c):
  xi = np.arange(65, 150)
  poly_norm = sum(poly3_nonorm(xi, a, b, c))
  
  return poly3_nonorm(x, a, b, c) + (-poly_norm + 1) / 85

def histogram(df):
  range_ = (65, 150)
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
  #factor = sum(MC)
  MC /= factor
  uncert /= factor

  return MC, uncert, bin_centers

def histogramBkg(df):
  MC, uncert, bin_centers = histogram(df)
  s = (bin_centers < 80) | (bin_centers > 100)
  return MC[s], uncert[s], bin_centers[s]

def plot(MC, uncert, bin_centers, f, popt, savepath, ratio_ylim, bonly=False):
  fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [3, 1]})

  x = np.linspace(65, 150, 500)

  axs[0].errorbar(bin_centers, MC, yerr=uncert, fmt="ko")
  axs[0].plot(x, f(x, *popt), label="Bern3 fit")

  axs[0].legend()
  axs[0].set_ylabel("Normaliised Events / (2.5 GeV)")

  ratio = MC / f(bin_centers, *popt)
  ratio_uncert = uncert / f(bin_centers, *popt)
  
  axs[1].errorbar(bin_centers, ratio, yerr=ratio_uncert, fmt="ko")
  axs[1].set_ylim(ratio_ylim)
  axs[1].axhline(1, color="k", linestyle="dashed")

  axs[1].set_xlabel(r"$m_{\gamma\gamma}$")
  axs[1].set_ylabel("MC / Fit")

  plt.savefig(savepath)

def plotSpB(MC, uncert, bin_centers, f, popt, savepath, ratio_ylim, bonly=False):
  fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [3, 1]})

  bkg_popt = popt.copy()
  bkg_popt[0] *= (1-bkg_popt[-1])
  bkg_popt[-1] = 0

  x = np.linspace(65, 150, 500)

  axs[0].errorbar(bin_centers, MC, yerr=uncert, fmt="ko")
  axs[0].plot(x, f(x, *popt), label="DCB + Bern3 fit")
  axs[0].plot(x, f(x, *bkg_popt), label="Bern3 component fit")

  axs[0].legend()
  axs[0].set_ylabel("Normaliised Events / (2.5 GeV)")

  ratio = MC / f(bin_centers, *popt)
  ratio_uncert = uncert / f(bin_centers, *popt)

  axs[1].errorbar(bin_centers, ratio, yerr=ratio_uncert, fmt="ko")
  axs[1].set_ylim(ratio_ylim)
  axs[1].axhline(1, color="k", linestyle="dashed")

  axs[1].set_xlabel(r"$m_{\gamma\gamma}$")
  axs[1].set_ylabel("MC / Fit")

  plt.savefig(savepath)

df = pd.read_parquet(sys.argv[1])
df_inv = pd.read_parquet(sys.argv[2])

"""
----------------------------------------------------------------------
"""
print("b fit to inverted")

MC, uncert, bin_centers = histogramBkg(df_inv)

bkg_f = lambda x, N, a, b, c: N*poly3(x, a, b, c)
p0 = [1, *np.random.uniform(-10, 10, 3)]
bounds = [[0.5, 2],
          [-10, 10],
          [-10, 10],
          [-10, 10]]
bounds = np.array(bounds).T

res = spo.curve_fit(bkg_f, bin_centers, MC, p0, uncert, bounds=bounds)
bkg_popt = res[0]
print(list(bkg_popt))

#MC, uncert, bin_centers = histogram(df_inv)
plot(MC, uncert, bin_centers, bkg_f, bkg_popt, "inverted_fit_bonly.png", (0.5, 1.5))

"""
----------------------------------------------------------------------
"""
print("s+b fit to inverted")
MC, uncert, bin_centers = histogram(df_inv)

spb_f = lambda x, N, a, b, c, mean, sigma, beta1, m1, beta2, m2, frac: N*(frac*dcb(x, mean, sigma, beta1, m1, beta2, m2) + (1-frac)*poly3(x, a, b, c))

p0 = [1, *bkg_popt[1:], 90, 3, 1, 2, 1, 2, 0.95]
print(p0)
bounds = [[0.5, 2],
          [-10, 10],
          [-10, 10],
          [-10, 10],
          [85, 95],
          [2, 4],
          [0.5, 5],
          [1.01, 10],
          [0.5, 5],
          [1.01, 10],
          [0.8, 0.99]]
bounds = np.array(bounds).T

res = spo.curve_fit(spb_f, bin_centers, MC, p0, uncert, bounds=bounds)
spb_popt = res[0]
print(list(spb_popt))

plotSpB(MC, uncert, bin_centers, spb_f, spb_popt, "inverted_fit_spb.png", (0.5, 1.5))

"""
----------------------------------------------------------------------
"""
print("s+b fit to inverted subtract dcb")

sig_popt = spb_popt.copy()
sig_popt[0] *= sig_popt[-1]
sig_popt[-1] = 1

bkg_popt = spb_popt.copy()
bkg_popt[0] *= (1-bkg_popt[-1])
bkg_popt[-1] = 0

MC_minus_DCB = MC - spb_f(bin_centers, *sig_popt)

plot(MC_minus_DCB, uncert, bin_centers, spb_f, bkg_popt, "inverted_fit_spb_subtract.png", (0.5, 1.5))

"""
----------------------------------------------------------------------
"""
print("b fit to veto")

MC, uncert, bin_centers = histogramBkg(df)

bkg_f = lambda x, N, a, b, c: N*poly3(x, a, b, c)
p0 = [1, *np.random.uniform(-10, 10, 3)]
print(p0)
bounds = [[0.5, 2],
          [-10, 10],
          [-10, 10],
          [-10, 10]]
bounds = np.array(bounds).T

res = spo.curve_fit(bkg_f, bin_centers, MC, p0, uncert, bounds=bounds)
bkg_popt = res[0]
print(list(bkg_popt))

MC, uncert, bin_centers = histogram(df)
plot(MC, uncert, bin_centers, bkg_f, bkg_popt, "veto_fit_bonly.png", (0.5, 1.5))

"""
----------------------------------------------------------------------
"""
print("spb fit to veto with inverted shape")

MC, uncert, bin_centers = histogram(df)

spb_f = lambda x, N, a, b, c, mean, sigma, beta1, m1, beta2, m2, frac: N*(frac*dcb(x, mean, sigma, beta1, m1, beta2, m2) + (1-frac)*poly3(x, a, b, c))

p0 = np.array([1, *bkg_popt[1:], 90, 3, 1, 2, 1, 2, 0.5])
bounds = [[0.5, 2],
          [-10, 10],
          [-10, 10],
          [-10, 10],
          [85, 95],
          [2, 4],
          [0.5, 5],
          [1.01, 10],
          [0.5, 5],
          [1.01, 10],
          [0.1, 0.9]]
bounds = np.array(bounds).T

p0[4:-1] = spb_popt[4:-1]
bounds[0, 4:-1] = p0[4:-1] - 0.000001
bounds[1, 4:-1] = p0[4:-1] + 0.000001

res = spo.curve_fit(spb_f, bin_centers, MC, p0, uncert, bounds=bounds)
spb_popt = res[0]
print(list(spb_popt))

plotSpB(MC, uncert, bin_centers, spb_f, spb_popt, "veto_fit_spb_inverted_dcb.png", (0.5, 1.5))

"""
----------------------------------------------------------------------
"""
print("spb fit to veto")
MC, uncert, bin_centers = histogram(df)

spb_f = lambda x, N, a, b, c, mean, sigma, beta1, m1, beta2, m2, frac: N*(frac*dcb(x, mean, sigma, beta1, m1, beta2, m2) + (1-frac)*poly3(x, a, b, c))

p0 = [1, *bkg_popt[1:], 90, 3, 1, 2, 1, 2, 0.5]
print(p0)
bounds = [[0.5, 2],
          [-10, 10],
          [-10, 10],
          [-10, 10],
          [85, 95],
          [2, 4],
          [0.5, 5],
          [1.01, 10],
          [0.5, 5],
          [1.01, 10],
          [0.1, 0.9]]
bounds = np.array(bounds).T

res = spo.curve_fit(spb_f, bin_centers, MC, p0, uncert, bounds=bounds)
spb_popt = res[0]
print(list(spb_popt))

plotSpB(MC, uncert, bin_centers, spb_f, spb_popt, "veto_fit_spb.png", (0.5, 1.5))