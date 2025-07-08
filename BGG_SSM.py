"""BGG_SSM.py
=================

State-space model for unsmoothing private‑equity NAV returns following
Brown, Ghysels and Gredil (2023).  The script introduces one lag of the
four factors in the observation equation and automatically selects the
subset of factors whose static regression t‑statistics are significant
(\|t\|\>1.96).  Dynamic betas and a rolling R² are saved and plotted.

The Excel file ``PE _data.xlsx`` in the same folder must provide the
columns ``Date`` (quarterly), ``PE - RF`` and the factor returns
``Mkt-RF``, ``SMB``, ``HML`` and ``Liq`` from 1984Q1 onward.

Running the script produces ``BGG_state_results.npz`` along with
``regression_results.csv`` and PNG charts.  Matplotlib uses the ``Agg``
backend so the script never blocks waiting for windows to close.
"""

from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")  # no GUI required
import matplotlib.pyplot as plt

DATA_FILE = "PE _data.xlsx"  # note the space as in the original file

# ── 1  load data ─────────────────────────────────────────────────────
if not Path(DATA_FILE).exists():
    raise FileNotFoundError(DATA_FILE)

df = pd.read_excel(DATA_FILE)
df = df.sort_values("Date").reset_index(drop=True)

dates = pd.to_datetime(df["Date"]).to_numpy("datetime64[D]")
pe_r = df["PE - RF"].to_numpy(float)  # excess PE return (reported)
F0 = np.column_stack([
    df["Mkt-RF"], df["SMB"], df["HML"], df["Liq"]
]).astype(float)
T = len(pe_r)
K = F0.shape[1]

# build one lag of each factor
F_lag = np.vstack([np.zeros((1, K)), F0[:-1]])
F_full = np.hstack([F0, F_lag])
FAC_NAMES = ["Mkt", "SMB", "HML", "LIQ", "Mkt_L1", "SMB_L1", "HML_L1", "LIQ_L1"]
KF = F_full.shape[1]

# ── 2  exhaustive regressions to choose factors ─────────────────────
pe_shift = np.concatenate(([0.0], pe_r[:-1]))
unsm_rough = (pe_r - 0.0*pe_shift)  # start with no smoothing
X_full = sm.add_constant(F_full[1:])
y_full = unsm_rough[1:]

rows = []
for k in range(1, KF+1):
    for idx in combinations(range(KF), k):
        X = sm.add_constant(F_full[1:, idx])
        res = sm.OLS(y_full, X).fit()
        tvals = res.tvalues[1:]
        if np.all(np.abs(tvals) >= 1.96):
            rows.append({
                "vars": ",".join([FAC_NAMES[i] for i in idx]),
                "adjR2": res.rsquared_adj,
                **{f"t_{FAC_NAMES[i]}": t for i, t in zip(idx, tvals)},
            })

reg_df = pd.DataFrame(rows)
reg_df.to_csv("regression_results.csv", index=False)

if reg_df.empty:
    raise ValueError("No factor combination has all coefficients significant")

best_row = reg_df.loc[reg_df["adjR2"].idxmax()]
keep_idx = [FAC_NAMES.index(v) for v in best_row["vars"].split(",")]
F_use = F_full[:, keep_idx]
K_use = F_use.shape[1]

# ── 3  initial OLS to seed state mean ────────────────────────────────
X_ols = sm.add_constant(F_use)
beta0 = np.linalg.lstsq(X_ols, pe_r, rcond=None)[0]
state_mean0 = beta0.copy()
state_cov0 = np.diag([0.1] + [1.0]*K_use)

# process noise: small random walk variance
Q = np.diag([0.005**2] + [0.05**2]*K_use)

# measurement noise variance initial guess
sigma_nu2 = np.var(pe_r - X_ols @ beta0)

# ── 3  Kalman filter helper ─────────────────────────────────────────-

def kalman_loglik(lam: float) -> float:
    x = state_mean0.copy()
    P = state_cov0.copy()
    ll = 0.0
    prev = 0.0
    for t in range(T):
        # prediction
        P = P + Q
        y = pe_r[t] - lam*prev
        H = (1-lam) * np.concatenate(([1.0], F_use[t]))
        S = H @ P @ H + sigma_nu2
        K = (P @ H) / S
        innov = y - H @ x
        x = x + K * innov
        P = P - np.outer(K, H) @ P
        ll += -0.5*(np.log(2*np.pi*S) + innov**2 / S)
        prev = pe_r[t]
    return ll

# ── 4  estimate smoothing parameter λ via ML ─────────────────────────
LAM_GRID = np.arange(0.0, 0.95, 0.01)
best_lambda = max(LAM_GRID, key=kalman_loglik)
print(f"Estimated lambda = {best_lambda:.3f}")

# ── 5  final filtering pass ─────────────────────────────────────────
alpha_path = np.zeros(T)
beta_path = np.zeros((T, K_use))
unsmoothed = np.zeros(T)

x = state_mean0.copy()
P = state_cov0.copy()
prev = 0.0
for t in range(T):
    P = P + Q
    y = pe_r[t] - best_lambda*prev
    H = (1-best_lambda) * np.concatenate(([1.0], F_use[t]))
    S = H @ P @ H + sigma_nu2
    K = (P @ H) / S
    innov = y - H @ x
    x = x + K * innov
    P = P - np.outer(K, H) @ P

    alpha_path[t] = x[0]
    beta_path[t] = x[1:]
    unsmoothed[t] = x[0] + x[1:] @ F_use[t]
    prev = pe_r[t]

# unsmoothed series including idiosyncratic component
pe_shift = np.concatenate(([0.0], pe_r[:-1]))
unsm_rough = (pe_r - best_lambda*pe_shift) / (1-best_lambda)

# rolling R^2 over 20 quarters
ROLL = 20
roll_R2 = np.full(T, np.nan)
for i in range(ROLL-1, T):
    y_win = unsmoothed[i-ROLL+1:i+1]
    X_win = sm.add_constant(F_use[i-ROLL+1:i+1])
    b = np.linalg.lstsq(X_win, y_win, rcond=None)[0]
    yhat = X_win @ b
    ss_res = np.sum((y_win - yhat)**2)
    ss_tot = np.sum((y_win - np.mean(y_win))**2)
    roll_R2[i] = 1 - ss_res/ss_tot

np.savez(
    "BGG_state_results.npz",
    dates=dates,
    pe_ret=pe_r,
    unsmoothed_series=unsmoothed,
    unsmoothed_rough=unsm_rough,
    alpha_path=alpha_path,
    beta_path=beta_path,
    roll_R2=roll_R2,
    lambda_est=best_lambda,
    )

# ── 6  figures (saved headlessly) ───────────────────────────────────
start = ROLL - 1
plt.figure(figsize=(8, 1.8*K_use))
for i in range(K_use):
    plt.plot(dates[start:], beta_path[start:, i], label=best_row["vars"].split(",")[i])
plt.axhline(0, color="black", linewidth=0.5)
plt.title("Dynamic betas")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("dynamic_betas.png", dpi=150)
plt.close()

plt.figure(figsize=(8,3))
plt.plot(dates[start:], roll_R2[start:])
plt.ylim(0, 1)
plt.title("Rolling R^2 (20 quarters)")
plt.tight_layout()
plt.savefig("rolling_R2.png", dpi=150)
plt.close()

print("Results saved to BGG_state_results.npz and regression_results.csv")

# ── 7  analytical output (scatter, hist, lag search) ─────────────────

# ----- Results.py functionality -----
df_res = pd.DataFrame({
    "Date": dates,
    "Realised": unsm_rough,
    "Fitted": unsmoothed,
})
df_res["Residual"] = df_res["Realised"] - df_res["Fitted"]
r2_full = 1 - df_res["Residual"].var() / df_res["Realised"].var()
print(f"Whole-sample R^2 = {r2_full:.3f}")

plt.figure(figsize=(4,3))
plt.scatter(df_res["Fitted"], df_res["Realised"], s=10, alpha=0.7)
lims = [df_res[["Fitted", "Realised"]].min().min(),
        df_res[["Fitted", "Realised"]].max().max()]
plt.plot(lims, lims, "--", lw=1)
plt.title("Realised vs Fitted")
plt.tight_layout()
plt.savefig("scatter_realised_vs_fitted.png", dpi=150)
plt.close()

plt.figure(figsize=(4,3))
resid = df_res["Residual"]
plt.hist(resid, bins=30, density=True, alpha=0.65)
mu, sd = resid.mean(), resid.std()
x = np.linspace(resid.min(), resid.max(), 200)
plt.plot(x, (1/(sd*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mu)/sd)**2), lw=1.5)
plt.title("Residual distribution")
plt.tight_layout()
plt.savefig("residual_hist.png", dpi=150)
plt.close()

plt.figure(figsize=(8,3))
plt.plot(dates, df_res["Realised"], label="Realised")
plt.plot(dates, df_res["Fitted"], label="Fitted")
plt.title("Realised vs Fitted")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("series_fit.png", dpi=150)
plt.close()

ROLL_B = 8
plt.figure(figsize=(8, 1.8*K_use))
for i in range(K_use):
    rb = pd.Series(beta_path[:, i]).rolling(ROLL_B).mean()
    plt.plot(dates, rb, label=best_row["vars"].split(",")[i])
plt.axhline(0, color="black", lw=0.5)
plt.title(f"Rolling betas ({ROLL_B}-qtr MA)")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("rolling_betas.png", dpi=150)
plt.close()

# ----- Plotting Script functionality -----
MAX_LAG = 6
factors = F0
sse = np.zeros(MAX_LAG+1)
adjR = np.zeros(MAX_LAG+1)
aic = np.zeros(MAX_LAG+1)
bic = np.zeros(MAX_LAG+1)
beta_store = []

for k in range(MAX_LAG+1):
    X_list = [np.ones(T)]
    for lag in range(k+1):
        lagged = np.roll(factors, lag, axis=0)
        lagged[:lag] = np.nan
        X_list.append(lagged)
    X = np.column_stack(X_list)
    valid = ~np.isnan(X).any(axis=1)
    y = unsm_rough[valid]
    Xv = X[valid]
    res = sm.OLS(y, Xv).fit()
    sse[k] = np.sum(res.resid**2)
    adjR[k] = res.rsquared_adj
    aic[k] = res.aic
    bic[k] = res.bic
    beta_store.append(res.params)

best_k = int(np.argmin(bic))
X_best_list = [np.ones(T)]
for lag in range(best_k+1):
    lagged = np.roll(factors, lag, axis=0)
    lagged[:lag] = np.nan
    X_best_list.append(lagged)
X_best = np.column_stack(X_best_list)
fitted = np.dot(X_best, beta_store[best_k])
fitted[np.isnan(fitted)] = np.nan

from matplotlib.gridspec import GridSpec
plt.figure(figsize=(13,8))
gs = GridSpec(2,3, height_ratios=[1,1.1])

ax1 = plt.subplot(gs[0,0])
ax1.plot(range(MAX_LAG+1), sse, marker="o")
ax1.set_xlabel("Num. Lags"); ax1.set_ylabel("Sum-sq Err")
ax1.set_title("Prediction Error")
ax1.axvline(best_k, color="g", ls="--")

ax2 = plt.subplot(gs[0,1])
for i, name in enumerate(["Mkt", "SMB", "HML", "LIQ"]):
    weights = [beta_store[j][1 + i*(j+1)] for j in range(MAX_LAG+1)]
    ax2.plot(range(MAX_LAG+1), weights, marker="o", label=name)
ax2.set_xlabel("Num. Lags"); ax2.set_ylabel("Weight")
ax2.set_title("Optimal Factor Weights")
ax2.legend(fontsize=8)

ax3 = plt.subplot(gs[0,2])
ax3.plot(range(MAX_LAG+1), adjR, marker="o")
ax3.set_xlabel("Num. Lags"); ax3.set_ylabel("Adj R^2")
ax3.set_title("Model Fit")
ax3.axvline(best_k, color="r", ls="--")

ax4 = plt.subplot(gs[1,0])
ax4.plot(range(MAX_LAG+1), aic, marker="o", label="AIC")
ax4.plot(range(MAX_LAG+1), bic, marker="s", label="BIC")
ax4.set_xlabel("Num. Lags"); ax4.set_ylabel("IC")
ax4.set_title("Model Selection")
ax4.legend()

ax5 = plt.subplot(gs[1,1:])
ax5.plot(dates, unsm_rough, label="Unsmooothed", lw=1)
ax5.plot(dates, fitted, "--", label="Smoothed (FOLB)", lw=1)
ax5.set_title("Factor-Mimicking Series")
ax5.legend()
plt.tight_layout()
plt.savefig("lag_diagnostics.png", dpi=150)
plt.close()
