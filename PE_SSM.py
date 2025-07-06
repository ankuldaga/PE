import numpy as np
import pandas as pd

# 1. Load the data from Excel
data = pd.read_excel("PE _data.xlsx")
# Ensure the data is sorted by date and set index if needed
data = data.sort_values("Date").reset_index(drop=True)
# Extract the relevant series
dates = data["Date"]
pe_ret = data["PE - RF"].values    # PE excess returns (reported, smoothed)
mkt = data["Mkt-RF"].values        # Market excess return
smb = data["SMB"].values           # SMB factor
hml = data["HML"].values           # HML factor
liq = data["Liq"].values           # Liquidity factor
T = len(pe_ret)  # number of quarters

# 2. Initialize state and parameters
# Initial state (alpha and betas) from a quick OLS on the first few years (or full sample as a guess)
X = np.column_stack([mkt, smb, hml, liq])
X = np.hstack([np.ones((T,1)), X])  # include intercept
# OLS estimate for a starting point
ols_coefs = np.linalg.lstsq(X, pe_ret, rcond=None)[0]  # [alpha, beta_M, beta_SMB, beta_HML, beta_LIQ]
alpha0, betaM0, betaSMB0, betaHML0, betaLIQ0 = ols_coefs

# State vector initial mean
state_mean = np.array([alpha0, betaM0, betaSMB0, betaHML0, betaLIQ0], dtype=float)
# Initial covariance: set relatively large uncertainty
state_cov = np.diag([0.1, 1.0, 1.0, 1.0, 1.0]) 
# (Variance of 0.1 for alpha and 1.0 for betas implies std dev ~0.32 and 1.0 respectively – quite diffuse)

# Process noise covariance Q: small values to allow slow drift
Q = np.diag([0.005**2, 0.05**2, 0.05**2, 0.05**2, 0.05**2])
# Here, we allow betas to vary a bit more (5% stdev per quarter) than alpha (0.5% per quarter).
# These can be tuned or estimated; small Q means the state changes slowly over time.

# Measurement noise variance (sigma_nu^2): initialize with a guess
# Use sample variance of residuals from OLS as starting guess for measurement noise
ols_resid = pe_ret - X.dot(ols_coefs)
sigma_nu2 = np.var(ols_resid)  # initial guess for observation noise variance

# 3. Define a function to run Kalman filter and return log-likelihood for a given lambda (and sigma_nu2, Q)
def run_kalman_filter(lambda_val, return_loglik=True):
    lam = lambda_val
    # Use global Q, sigma_nu2, state_mean, state_cov as initialized for each run
    # (Alternatively, could also treat sigma_nu2 as variable to optimize, but here we fix or grid-search it)
    x_est = state_mean.copy()
    P_est = state_cov.copy()
    log_lik = 0.0
    # Loop through time
    prev_obs = 0.0  # we will set prev_obs for t=0; since we have no t=-1, we assume r_{-1}=0 for initialization
    for t in range(T):
        # Prediction step
        # (Since F = I, predicted state = x_est (no change), and P_pred = P_est + Q)
        x_pred = x_est
        P_pred = P_est + Q
        # Observation prediction
        # Compute y_t = r_obs_t - lam * r_obs_{t-1}
        # For t=0, we take prev_obs = 0 (assume no prior return, or treat first obs as partially unsmoothed).
        y_t = pe_ret[t] - lam * prev_obs
        # Compute H_t = (1-lam) * [1, factors_t]
        H_t = (1 - lam) * np.array([1.0, mkt[t], smb[t], hml[t], liq[t]], dtype=float)
        # Predicted observation: y_pred = H_t * x_pred
        y_pred = H_t.dot(x_pred)
        # Innovation: error between actual and predicted
        innov = y_t - y_pred
        # Innovation variance: S_t = H P_pred H^T + sigma_nu^2
        S_t = H_t.dot(P_pred).dot(H_t.T) + sigma_nu2
        # Kalman Gain
        K_t = P_pred.dot(H_t.T) / S_t  # (5x1 vector)
        # State update
        x_est = x_pred + K_t * innov
        # Covariance update
        P_est = P_pred - np.outer(K_t, H_t).dot(P_pred)
        # Update log-likelihood
        if return_loglik:
            log_lik += -0.5*(np.log(2*np.pi*S_t) + (innov**2)/S_t)
        # Set prev_obs for next iteration (now current obs becomes previous)
        prev_obs = pe_ret[t]
    return (log_lik if return_loglik else (x_est, P_est))

# 4. Grid search to estimate lambda (and possibly sigma_nu2)
best_ll = -np.inf
best_lambda = None
# We can loop over plausible lambda values (0 = no smoothing, up to ~0.9 heavy smoothing)
for lam in np.linspace(0.0, 0.9, 19):  # step 0.05
    ll = run_kalman_filter(lam)
    if ll > best_ll:
        best_ll = ll
        best_lambda = lam

# Refine search around best_lambda with finer resolution
if best_lambda is not None:
    lam_center = best_lambda
    search_range = np.linspace(max(0, lam_center-0.05), min(0.99, lam_center+0.05), 11)
    for lam in search_range:
        ll = run_kalman_filter(lam)
        if ll > best_ll:
            best_ll = ll
            best_lambda = lam

print(f"Estimated smoothing parameter lambda = {best_lambda:.3f}")
# (We expect lambda to be significantly >0, indicating noticeable smoothing.)

# 5. With estimated lambda, run Kalman filter to get state estimates and de-smoothed returns
lam = best_lambda if best_lambda is not None else 0.0  # fallback to 0 if not found
x_est = state_mean.copy()
P_est = state_cov.copy()
prev_obs = 0.0
# Arrays to store results
alphas = np.zeros(T)
betas_M = np.zeros(T); betas_SMB = np.zeros(T)
betas_HML = np.zeros(T); betas_LIQ = np.zeros(T)
true_ret_est = np.zeros(T)  # to store estimated true return (expected value)
for t in range(T):
    # Prediction
    x_pred = x_est
    P_pred = P_est + Q
    # Observation
    y_t = pe_ret[t] - lam * prev_obs
    H_t = (1 - lam) * np.array([1.0, mkt[t], smb[t], hml[t], liq[t]])
    y_pred = H_t.dot(x_pred)
    innov = y_t - y_pred
    S_t = H_t.dot(P_pred).dot(H_t.T) + sigma_nu2
    K_t = P_pred.dot(H_t.T) / S_t
    # Update state
    x_est = x_pred + K_t * innov
    P_est = P_pred - np.outer(K_t, H_t).dot(P_pred)
    # Store state and compute true return estimate
    alphas[t]    = x_est[0]
    betas_M[t]   = x_est[1];  betas_SMB[t] = x_est[2]
    betas_HML[t] = x_est[3];  betas_LIQ[t] = x_est[4]
    # Expected true excess return = alpha + sum(beta * factor)
    true_ret_est[t] = x_est[0] + x_est[1]*mkt[t] + x_est[2]*smb[t] + x_est[3]*hml[t] + x_est[4]*liq[t]
    # Update prev_obs
    prev_obs = pe_ret[t]

# Also compute an alternative "unsmoothed return" series including idiosyncratic component:
unsmoothed_series = np.zeros(T)
unsmoothed_series[0] = pe_ret[0] / (1 - lam)  # for first period, assuming prev_obs=0
for t in range(1, T):
    unsmoothed_series[t] = (pe_ret[t] - lam * pe_ret[t-1]) / (1 - lam)

# 6. Examine the results: compare volatility and autocorrelation of original vs de-smoothed returns
orig_vol = np.std(pe_ret)
unsm_vol = np.std(unsmoothed_series)
orig_ac1 = pd.Series(pe_ret).autocorr(lag=1)
unsm_ac1 = pd.Series(unsmoothed_series).autocorr(lag=1)
print(f"Reported PE Return: Volatility={orig_vol:.3f}, AR(1)={orig_ac1:.3f}")
print(f"De-smoothed PE Return: Volatility={unsm_vol:.3f}, AR(1)={unsm_ac1:.3f}")

# (We expect de-smoothed volatility >> reported volatility, and AR(1) near 0 or negative for de-smoothed.)
# Finally, show first few estimates as a sanity check
#for i in range(len(dates)):
for i in range(3):
    print(f"Q{dates[i].quarter}-'{str(dates[i].year)[-2:]}: Reported={pe_ret[i]:+.3f}, De-smoothed≈{unsmoothed_series[i]:+.3f}")

#charting the results
import matplotlib
matplotlib.use("TkAgg")      # or "QtAgg", "WXAgg" … any GUI backend you have
# ↑ MUST come before the first pyplot import
import matplotlib.pyplot as plt
plt.figure()
plt.plot(dates, pe_ret, label='Reported', linewidth=1)
plt.plot(dates, unsmoothed_series, label='De-smoothed', linewidth=1)
plt.legend(); plt.title('PE Reported vs De-smoothed'); plt.show()
plt.show(block=False)        # window stays open, but code keeps running

np.savez("PE_state_arrays.npz",
         dates=dates.values.astype('datetime64[D]'),
         pe_ret           = pe_ret,              # reported series
         unsmoothed_series = unsmoothed_series,   # true (realised)
         mkt = mkt, smb = smb, hml = hml, liq = liq,
         alphas=alphas,
         betas_M=betas_M, betas_SMB=betas_SMB,
         betas_HML=betas_HML, betas_LIQ=betas_LIQ,
         true_ret_est=true_ret_est)
print("✅ PE_SSM_finished")   # this now prints immediately
# end of PE_SSM.py