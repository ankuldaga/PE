# # PE_SSM_significant_only.py — automatic factor‑subset selection
# """Pipeline
# =========
# This script performs automatic factor‑subset selection and NAV unsmoothing for
# private‑equity returns.

# Inputs
# ------
# * **PE _data.xlsx** – Excel file in the same folder. Required columns:
#   `Date`, `PE - RF`, `Mkt-RF`, `SMB`, `HML`, `Liq`.

# Outputs
# -------
# * **subset_results.csv** – t‑stats, adj R², and significance flag for each of
#   the 15 factor combinations.
# * **pe_vs_true.png**      – reported vs de‑smoothed return plot (headless safe).
# * **PE_state_arrays.npz** – NumPy bundle with dates, reported & unsmoothed
#   series, α(t), βₖ(t), kept factor names, and λ̂.  Suitable for Results.py and
#   Plotting Script.py.

# Algorithm
# ---------
# 1.  Pre‑estimate λ̂ using all four factors.
# 2.  Build rough unsmoothed returns  r̃ₜ = (r_obs − λ̂·r_obs[‑1])/(1−λ̂).
# 3.  Exhaustively regress r̃ on every non‑empty subset of factors.  Keep only
#     models where **all β’s have |t| ≥ 1.96**; pick the one with the highest
#     adjusted R².
# 4.  Run a Kalman filter with that subset to obtain final λ̂, α(t), βₖ(t), and
#     unsmoothed returns.

# Headless/CI‑safe: `matplotlib.use('Agg')` so code never blocks on `plt.show()`.
# Requires SciPy ≤ 1.15.x, statsmodels ≥ 0.14, NumPy ≥ 1.25.
# """ SciPy <= 1.15.x, statsmodels >= 0.14, NumPy >= 1.25.

from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── constants ───────────────────────────────────────────────────────
DATA_FILE = "PE _data.xlsx"        # original file (note leading space)
SIG_Z     = 1.96                   # 95 % two‑sided threshold
LAM_GRID  = np.arange(0.0, 0.95, 0.05)
FACT_NAMES = np.array(["Mkt", "SMB", "HML", "LIQ"])

# ── 1  load NAV & factor data ───────────────────────────────────────
if not Path(DATA_FILE).exists():
    raise FileNotFoundError(DATA_FILE)

df   = pd.read_excel(DATA_FILE).sort_values("Date").reset_index(drop=True)
pe_r = df["PE - RF"].to_numpy(float)
F_all = np.column_stack([
    df["Mkt-RF"], df["SMB"], df["HML"], df["Liq"]
]).astype(float)

dates = pd.to_datetime(df["Date"]).to_numpy("datetime64[D]")
T     = len(pe_r)

# ── 2  quick λ̂ with all factors (static) ───────────────────────────
X_full = sm.add_constant(F_all)
beta0  = np.linalg.lstsq(X_full, pe_r, rcond=None)[0]
sig2   = np.var(pe_r - X_full @ beta0)

# simple LL under random‑walk state but static β good enough for λ pre‑estimate

def ll_static(lam: float) -> float:
    prev, ll = 0.0, 0.0
    for t in range(T):
        yhat = (1-lam)*X_full[t]@beta0 + lam*prev
        ll  += -0.5*(np.log(2*np.pi*sig2) + (pe_r[t]-yhat)**2/sig2)
        prev = pe_r[t]
    return ll

lam0 = max(LAM_GRID, key=ll_static)
print(f"Pre‑estimate λ̂ = {lam0:.3f}\n")

# ── 3  rough unsmoothed series ──────────────────────────────────────
pe_shift   = np.concatenate(([0.0], pe_r[:-1]))
unsm_rough = (pe_r - lam0*pe_shift) / (1-lam0)

# ── 4  exhaustive subset regressions ────────────────────────────────
subset_rows = []
for k in range(1, 5):                      # non‑empty subsets of 4 factors
    for idx in combinations(range(4), k):
        cols = list(idx)
        X    = sm.add_constant(F_all[:, cols])
        res  = sm.OLS(unsm_rough, X).fit()
        tvals = res.tvalues[1:]            # skip intercept
        sig   = np.all(np.abs(tvals) >= SIG_Z)
        subset_rows.append({
            "subset"      : ",".join(FACT_NAMES[list(cols)]),
            "k"           : k,
            "adjR2"       : res.rsquared_adj,
            "sig_all"     : sig,
            **{f"t_{FACT_NAMES[c]}": t for c, t in zip(cols, tvals)},
        })

sub_df = pd.DataFrame(subset_rows)
sub_df.to_csv("subset_results.csv", index=False)
print("Saved subset_results.csv (all combo t‑stats + adjR²)\n")

sig_df = sub_df[sub_df["sig_all"]].copy()
if sig_df.empty:
    raise ValueError("No factor combination has all coefficients significant at 95 %.")

best_row  = sig_df.loc[sig_df["adjR2"].idxmax()]
keep_cols = best_row["subset"].split(",")
keep_mask = np.isin(FACT_NAMES, keep_cols)
print("Chosen model:", keep_cols, "with adjR² =", round(best_row["adjR2"],3),"\n")

F = F_all[:, keep_mask]
K = F.shape[1]

# ── 5  Kalman filter with chosen factors ────────────────────────────
Xk     = sm.add_constant(F)
beta_i = np.linalg.lstsq(Xk, pe_r, rcond=None)[0]
mu_s   = beta_i.copy()
P_s    = np.diag([0.1]+[1.0]*K)
Q_s    = np.diag([0.005**2]+[0.05**2]*K)
sig2   = np.var(pe_r - Xk@beta_i)

# log‑likelihood under RW‑state model
def ll(lam: float) -> float:
    x, P, ll_, prev = mu_s.copy(), P_s.copy(), 0.0, 0.0
    for t in range(T):
        P += Q_s; y = pe_r[t] - lam*prev
        H = (1-lam)*np.concatenate(([1.0], F[t]))
        S = H@P@H + sig2
        K = (P@H)/S
        innov = y - H@x
        x += K*innov; P -= np.outer(K,H)@P
        ll_ += -0.5*(np.log(2*np.pi*S)+innov**2/S)
        prev = pe_r[t]
    return ll_

lam_best = max(LAM_GRID, key=ll)
print(f"Final λ̂ = {lam_best:.3f}\n")

# final filtering pass
a_path = np.zeros(T); b_path = np.zeros((T,K)); unsm = np.zeros(T)
prev=0.0; x=mu_s.copy(); P=P_s.copy()
for t in range(T):
    P+=Q_s; y=pe_r[t]-lam_best*prev; H=(1-lam_best)*np.concatenate(([1.0],F[t]))
    S=H@P@H+sig2; K=(P@H)/S; innov=y-H@x; x+=K*innov; P-=np.outer(K,H)@P
    a_path[t]=x[0]; b_path[t]=x[1:]; unsm[t]=x[0]+x[1:]@F[t]; prev=pe_r[t]

# ── 6  diagnostics & figures ───────────────────────────────────── ─────────────────────────────────────
print(f"Reported σ={np.std(pe_r):.3f}, AR1={pd.Series(pe_r).autocorr():.3f}
"
      f"Unsmoothed σ={np.std(unsm):.3f}, AR1={pd.Series(unsm).autocorr():.3f}")
# 6‑a  reported vs true
plt.figure(figsize=(8,4))
plt.plot(dates, pe_r, lw=1, label="Reported")
plt.plot(dates, unsm, lw=1, label="De‑smoothed")
plt.title("Reported vs De‑smoothed PE")
plt.legend(); plt.tight_layout(); plt.savefig("pe_vs_true.png", dpi=150); plt.close()

# 6‑b  dynamic β paths (≥ 10 years ≈ 40 quarters)
if K > 0:
    plt.figure(figsize=(9, 1.8*K))
    start_idx = 40 if T > 40 else 0          # skip first 10 yrs so betas have enough history
    for i, fac in enumerate(keep_cols):
        ax = plt.subplot(K, 1, i+1)
        ax.plot(dates[start_idx:], b_path[start_idx:, i], lw=1)
        ax.axhline(0, color="black", linewidth=0.4)
        ax.set_ylabel(f"β_{fac}")
        if i == 0:
            ax.set_title("Dynamic betas (10‑year window and beyond)")
    plt.tight_layout(); plt.savefig("dynamic_betas.png", dpi=150); plt.close()

# 6‑c  rolling 10‑year adj R²  (uses unsmoothed series)
roll_R2 = np.full(T, np.nan)
window  = 40
for t in range(window-1, T):
    y_win = unsm[t-window+1:t+1]
    X_win = sm.add_constant(F[t-window+1:t+1])
    res   = sm.OLS(y_win, X_win).fit()
    roll_R2[t] = res.rsquared_adj

plt.figure(figsize=(8,3))
plt.plot(dates[window-1:], roll_R2[window-1:], lw=1)
plt.ylim(0, 1)
plt.title("Rolling 10‑Year Adjusted R²")
plt.tight_layout(); plt.savefig("rolling_R2.png", dpi=150); plt.close()

# ── 7  save outputs ──────────────────────────────────────────────── ────────────────────────────────────────────────
np.savez("PE_state_arrays.npz", dates=dates, pe_ret=pe_r, unsmoothed_series=unsm,
         alpha=a_path, betas=b_path, kept_names=np.array(keep_cols), lambda_est=lam_best)
print("✅ Finished. Arrays + subset_results.csv + pe_vs_true.png generated\n")
