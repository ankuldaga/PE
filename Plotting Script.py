#Plotting Script
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from matplotlib.gridspec import GridSpec   # nicer layout

# ---------- load everything needed ----------
arr = np.load("PE_state_arrays.npz")

dates            = pd.to_datetime(arr["dates"])
unsmoothed_series= arr["unsmoothed_series"]
mkt, smb, hml, liq = arr["mkt"], arr["smb"], arr["hml"], arr["liq"]
# --------------------------------------------

# Settings
MAX_LAG = 6          # replicate your MATLAB range 0…6
target  = unsmoothed_series        # what we want to explain
factors = np.column_stack([mkt, smb, hml, liq])
fac_names = ["Mkt", "SMB", "HML", "LIQ"]
T = len(target)

# Containers for diagnostics
sse   = np.zeros(MAX_LAG+1)
adjR2 = np.zeros(MAX_LAG+1)
aic   = np.zeros(MAX_LAG+1)
bic   = np.zeros(MAX_LAG+1)
beta_store = []

# ---------------------------------------------------------------------
# Loop over number of lags k = 0…MAX_LAG
for k in range(MAX_LAG+1):
    # Build X_t = [1, factors_t, factors_{t-1}, … factors_{t-k}]
    X_list = [np.ones(T)]
    for lag in range(k+1):
        lagged_factors = np.roll(factors, lag, axis=0)
        lagged_factors[:lag, :] = np.nan        # first ‘lag’ rows undefined
        X_list.append(lagged_factors)
    X = np.column_stack(X_list)

    # Drop initial rows with NaN (due to lagging)
    valid_idx = ~np.isnan(X).any(axis=1)
    y = target[valid_idx]
    X = X[valid_idx]

    # OLS with statsmodels
    model = sm.OLS(y, X)
    res   = model.fit()

    sse[k]   = np.sum(res.resid**2)
    adjR2[k] = res.rsquared_adj
    aic[k]   = res.aic
    bic[k]   = res.bic
    beta_store.append(res.params)

# ---------------------------------------------------------------------
# Choose best lag by BIC  (could use AIC instead)
best_k = int(np.argmin(bic))
best_beta = beta_store[best_k]

# Re-compute fitted (factor-mimicking) series with best_k lags
X_best_list = [np.ones(T)]
for lag in range(best_k+1):
    lagged_factors = np.roll(factors, lag, axis=0)
    lagged_factors[:lag, :] = np.nan
    X_best_list.append(lagged_factors)
X_best = np.column_stack(X_best_list)
fitted = np.dot(X_best, best_beta)
fitted[np.isnan(fitted)] = np.nan     # for plotting

# ---------------------------------------------------------------------
# -------- Plotting   (mimics your MATLAB layout) ----------------------
plt.figure(figsize=(13, 8))
gs = GridSpec(2, 3, height_ratios=[1, 1.1])

# 1. Prediction error
ax1 = plt.subplot(gs[0, 0])
ax1.plot(range(MAX_LAG+1), sse, marker='o')
ax1.set_xlabel('Num. Lags'); ax1.set_ylabel('Sum-sq Prediction Error')
ax1.set_title('Prediction Error')
ax1.axvline(best_k, color='g', linestyle='--')
ax1.text(best_k, max(sse)*0.98, f'  best={best_k}', color='g', va='top')

# 2. Optimal factor weights
ax2 = plt.subplot(gs[0, 1])
for i, f in enumerate(fac_names):
    weights = [beta_store[k][1+i] if k>=0 else np.nan for k in range(MAX_LAG+1)]
    # weights above pulls only the contemporaneous coef; next lines pull lagged coefs too
    weights = []
    for k in range(MAX_LAG+1):
        # grab column indices for this factor across lags
        beta_k = beta_store[k]
        # first param is intercept, then factors (k+1)*(num_factors)
        w = [beta_k[1 + j + i* (k+1)] for j in range(0, (k+1))]  # j is lag index
        # take sum of absolute contributions or the 0-lag weight? choose 0-lag weight for clarity
        weights.append(w[0])
    ax2.plot(range(MAX_LAG+1), weights, marker='o', label=f)
ax2.set_xlabel('Num. Lags'); ax2.set_ylabel('Weight')
ax2.set_title('Optimal Factor Weights')
ax2.legend(loc='upper right', fontsize=8)

# 3. Model fit (Adj R²)
ax3 = plt.subplot(gs[0, 2])
ax3.plot(range(MAX_LAG+1), adjR2, marker='o')
ax3.set_xlabel('Num. Lags'); ax3.set_ylabel('Adj R²')
ax3.set_title('Model Fit (Adj R²)')
ax3.axvline(best_k, color='r', linestyle='--')
ax3.text(best_k, max(adjR2)*0.98, f'  best={best_k}', color='r', va='top')

# 4. Model Selection: AIC vs BIC
ax4 = plt.subplot(gs[1, 0])
ax4.plot(range(MAX_LAG+1), aic, marker='o', label='AIC')
ax4.plot(range(MAX_LAG+1), bic, marker='s', label='BIC')
ax4.set_xlabel('Num. Lags'); ax4.set_ylabel('Information Crit.')
ax4.set_title('Model Selection: AIC vs BIC')
ax4.legend()

# 5. Factor-mimicking return series
ax5 = plt.subplot(gs[1, 1:])
ax5.plot(dates, target, label='Unsmooothed', linewidth=1)
ax5.plot(dates, fitted, '--', label='Smoothed (FOLB)', linewidth=1)
ax5.set_title('Factor-Mimicking Return Series')
ax5.set_ylabel('Return')
ax5.legend()

plt.tight_layout()
plt.show()
