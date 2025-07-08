# State Space Model for PE NAV Desmoothing

This memo outlines the NAV-based state–space model used to unsmooth quarterly private equity (PE) returns. The specification follows Brown, Ghysels and Gredil (2023) and allows for time–varying alpha and factor betas.

## Model
Let $r_t$ denote the true excess return on the PE fund in quarter $t$ and let $r_t^{\text{obs}}$ be the reported (smoothed) return. Let $f_t$ be a vector of factor returns (market, SMB, HML, LIQ). The model is

$$
\begin{aligned}
 r_t &= \alpha_t + \beta_t' f_t + \varepsilon_t , & \varepsilon_t &\sim \mathcal{N}(0, \sigma_{\varepsilon}^2) ,\\
 r_t^{\text{obs}} &= \lambda r_{t-1}^{\text{obs}} + (1-\lambda) r_t + \nu_t , & \nu_t &\sim \mathcal{N}(0, \sigma_{\nu}^2) .
\end{aligned}
$$
The reported return is a geometrically smoothed version of the true return with smoothing parameter $\lambda\in[0,1)$. Setting $\lambda=0$ gives no smoothing.

The state vector collects the time–varying intercept and betas
$$
 x_t = \begin{bmatrix} \alpha_t \\ \beta_{M,t} \\ \beta_{\text{SMB},t} \\ \beta_{\text{HML},t} \\ \beta_{\text{LIQ},t} \end{bmatrix}.
$$
We assume a random–walk transition
$$
 x_t = x_{t-1} + w_t , \qquad w_t \sim \mathcal{N}(0, Q) ,
$$
where $Q$ is diagonal and governs the smoothness of the coefficients.

Combining the two equations yields the observation equation for the Kalman filter. Define
$$y_t = r_t^{\text{obs}} - \lambda r_{t-1}^{\text{obs}} , \qquad H_t = (1-\lambda)\begin{bmatrix}1 & f_t'\end{bmatrix} .$$
Then
$$
 y_t = H_t x_t + u_t , \qquad u_t \sim \mathcal{N}(0, \sigma_{\nu}^2) .
$$
This is a standard linear Gaussian state–space model which can be estimated via maximum likelihood using the Kalman filter.

## Estimation
1. **Initialization** – Obtain OLS estimates of $\alpha$ and $\beta$ on the full sample as starting values for $x_0$. The initial covariance is set to a diffuse diagonal matrix.
2. **Likelihood maximization** – For a grid of candidate $\lambda$ values (e.g. 0 to 0.95), run the Kalman filter to compute the log likelihood. The value maximizing the likelihood provides the smoothing estimate $\hat\lambda$.
3. **Filtering** – Using $\hat\lambda$, run the Kalman filter once more to generate filtered states $\hat{x}_t$ and to produce an unsmoothed return series
   $$ \hat r_t = \hat\alpha_t + \hat\beta_t' f_t . $$
   An alternative unsmoothed series including measurement noise is
   $$ \tilde r_t = \frac{r_t^{\text{obs}} - \hat\lambda r_{t-1}^{\text{obs}}}{1-\hat\lambda} . $$

The volatility and autocorrelation of the unsmoothed series can be compared with the reported series to verify that smoothing has been removed.

## Extensions implemented in `BGG_SSM.py`

* One lag of each factor is included in the observation equation.
* Prior to filtering, all possible factor subsets (including lags) are
  regressed on a rough unsmoothed series and kept only if every
  coefficient is significant ($|t|\ge1.96$).  The subset with the highest
  adjusted $R^2$ is chosen for the state‑space model and the regression
  results are saved to ``regression_results.csv``.
* Dynamic betas are extracted from the Kalman filter.  Rolling
  $R^2$ statistics over a 20‑quarter window are computed, and both the
  betas and the $R^2$ path are saved to PNG charts without opening GUI
  windows (``matplotlib`` uses the ``Agg`` backend).
