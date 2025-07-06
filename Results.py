# ------------------------------------------------------------------
#  AFTER your Kalman-filter loop has filled the following arrays:
#  dates, alphas, betas_M, betas_SMB, betas_HML, betas_LIQ,
#  true_ret_est  (unsmoothed expected true returns, i.e. fitted mean),
#  unsmoothed_series  (unsmoothed realised returns incl. idiosyncratic)
# ------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
plt.rcParams["figure.dpi"] = 110         # sharper plots
# Load the arrays from the saved file
arr = np.load("PE_state_arrays.npz")

pe_ret = arr["pe_ret"]  # reported PE returns
dates = arr["dates"]
alphas = arr["alphas"]
betas_M = arr["betas_M"]
betas_SMB = arr["betas_SMB"]
betas_HML = arr["betas_HML"]
betas_LIQ = arr["betas_LIQ"]
true_ret_est = arr["true_ret_est"]
unsmoothed_series = arr["unsmoothed_series"]

#charting the results
import matplotlib.pyplot as plt
plt.figure()
plt.plot(dates, pe_ret, label='Reported', linewidth=1)
plt.plot(dates, unsmoothed_series, label='De-smoothed', linewidth=1)
plt.legend(); plt.title('PE Reported vs De-smoothed'); plt.show()

# 1. Assemble a tidy DataFrame with betas, fitted, residual
df = pd.DataFrame({
    "Date":            dates,
    "Alpha":           alphas,
    "Beta_Mkt":        betas_M,
    "Beta_SMB":        betas_SMB,
    "Beta_HML":        betas_HML,
    "Beta_LIQ":        betas_LIQ,
    "True_Fitted":     true_ret_est,            # expected true return
    "True_Realised":   unsmoothed_series,       # realised unsmoothed
})
df["Residual"] = df["True_Realised"] - df["True_Fitted"]

# 2. Whole-sample R² (explained variation / total variation)
r_squared = 1 - df["Residual"].var() / df["True_Realised"].var()
print(f"Whole-sample R² of factor model (unsmoothed returns) = {r_squared:.3%}")

# 3. Save betas & residuals if you like
# df.to_excel("PE_betas_residuals.xlsx", index=False)

# 4. -----------  CHARTS  ----------------------------------------

# Figure 1 – Realised vs fitted (scatter + 45°)
plt.figure()
plt.scatter(df["True_Fitted"], df["True_Realised"], s=10, alpha=0.6)
lims = [plt.xlim()[0], plt.xlim()[1]]
plt.plot(lims, lims, linestyle="--", linewidth=1)
plt.title("Quarterly Uns­moothed PE Return:\nRealised vs Factor-Fitted")
plt.xlabel("Fitted true return")
plt.ylabel("Realised true return")
plt.tight_layout()
plt.show()

# Figure 2 – Histogram of residuals with normal PDF overlay
plt.figure()
resid = df["Residual"]
plt.hist(resid, bins=30, density=True, alpha=0.65, label="Residuals")
mu, sigma = resid.mean(), resid.std()
x = np.linspace(resid.min(), resid.max(), 200)
#plt.plot(x, sm.distributions.ECDF(resid).y, alpha=0) # ensures statsmodels imported
plt.plot(x, (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mu)/sigma)**2),
         linewidth=1.5, label="Normal PDF")
plt.title("Residual Distribution\n(unsmoothed return minus factor fit)")
plt.legend(); plt.tight_layout(); plt.show()

# Figure 3 – Time-series fit
plt.figure()
plt.plot(df["Date"], df["True_Realised"], label="Realised", linewidth=1)
plt.plot(df["Date"], df["True_Fitted"], label="Fitted", linewidth=1)
plt.title("Uns­moothed PE Return – Realised vs Fitted")
plt.legend(); plt.tight_layout(); plt.show()

# Figure 4 – Rolling market beta (8-quarter window)
window = 8
rolling_betaM = df["Beta_Mkt"].rolling(window).mean()
plt.figure()
plt.plot(df["Date"], rolling_betaM, linewidth=1)
plt.axhline(0, color="black", linewidth=0.5)
plt.title(f"Rolling Market β ( {window}-quarter MA )")
plt.tight_layout(); plt.show()
