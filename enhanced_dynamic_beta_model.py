"""
Enhanced Dynamic Beta Model with AR-Smoothed Betas and Sandwich Standard Errors
================================================================================

This module extends the MMDA Dynamic Beta Model with two critical improvements
based on practitioner feedback:

1. AUTOREGRESSIVE BETA SMOOTHING:
   The original model computes beta as a deterministic function of rate level
   and volatility, which can produce overly volatile period-to-period beta changes.
   This enhancement adds an AR(1) smoothing structure:
   
       β̃_t = β_t^*                            if |β_t^* - β̃_{t-1}| ≤ k
       β̃_t = β̃_{t-1} + k · sign(β_t^* - β̃_{t-1})  otherwise
   
   where β_t^* is the "unconstrained" S-curve + volatility beta, β̃_t is the
   smoothed (constrained) beta, and k is a user-supplied tolerance controlling
   the maximum allowable period-to-period change.
   
   This produces betas that are economically smooth while preserving asymptotic
   convergence to the equilibrium S-curve.

2. SANDWICH (ROBUST) MLE STANDARD ERRORS:
   The original model uses either (a) no SEs for nonlinear parameters, or
   (b) Newey-West HAC for linearized parameters only. Both approaches are
   incomplete. With time-series data, MLE observations are NOT independent,
   so the usual Hessian-based variance formula V = H^{-1} is inconsistent.
   
   The correct asymptotic covariance for MLE with dependent observations uses
   the Huber-White sandwich estimator:
   
       V_sandwich = H^{-1} · S · H^{-1}
   
   where:
       H = (1/n) Σ ∂²ℓ/∂θ∂θ' evaluated at θ̂  (observed information)
       S = (1/n) Σ_t Σ_s g_t · g_s'           (long-run variance of scores)
       g_t = ∂ℓ_t/∂θ evaluated at θ̂            (individual score contributions)
   
   S incorporates Newey-West HAC weighting to account for serial correlation
   in the score vectors.

Model Owner: Chih L. Chen
Date: February 2026
Version: 2.0

References:
- White, H. (1982). "Maximum likelihood estimation of misspecified models."
  Econometrica, 50(1), 1-25.
- Newey, W.K. and West, K.D. (1987). "A simple, positive semi-definite, 
  heteroskedasticity and autocorrelation consistent covariance matrix."
  Econometrica, 55(3), 703-708.
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# AR-SMOOTHED BETA FUNCTIONS
# =============================================================================

def logistic_beta(r, k, m, beta_min, beta_max):
    """Standard logistic S-curve beta function."""
    exp_term = np.exp(-k * (r - m))
    beta = beta_min + (beta_max - beta_min) / (1 + exp_term)
    return np.clip(beta, beta_min, beta_max)


def asymmetric_volatility_beta(r, k, m, beta_min, beta_max, vol_ratio,
                                lambda_up, lambda_down, rate_change):
    """
    Asymmetric volatility-adjusted beta (unconstrained).
    
    β_t^* = β^{level}(r_t) × (1 - λ_eff × vol_ratio_t)
    where λ_eff = λ_up if Δr > 0, else λ_down.
    """
    base_beta = logistic_beta(r, k, m, beta_min, beta_max)
    lambda_effective = np.where(rate_change > 0, lambda_up, lambda_down)
    vol_adjustment = np.clip(1 - lambda_effective * vol_ratio, 0.5, 1.5)
    adjusted_beta = base_beta * vol_adjustment
    return np.clip(adjusted_beta, beta_min * 0.8, beta_max * 1.2)


def ar_smoothed_beta(beta_unconstrained, max_delta_k, beta_init=None):
    """
    Apply autoregressive smoothing constraint to a beta time series.
    
    Enforces |β̃_t - β̃_{t-1}| ≤ k (user-supplied tolerance) by clamping
    changes that exceed the tolerance while allowing the smoothed beta to
    gradually converge to the unconstrained value.
    
    Parameters
    ----------
    beta_unconstrained : array-like, shape (T,)
        The "target" beta series from the S-curve + volatility model.
    max_delta_k : float
        Maximum allowable absolute change in beta per period.
        Typical values: 0.01-0.05 (i.e., 1-5 percentage points per month).
        Set to np.inf to disable smoothing (recover original model).
    beta_init : float, optional
        Starting value for the smoothed series. If None, uses first value
        of beta_unconstrained.
    
    Returns
    -------
    beta_smoothed : ndarray, shape (T,)
        Smoothed beta series satisfying the AR constraint.
    
    Notes
    -----
    The smoothing operation is:
        β̃_t = β̃_{t-1} + clip(β_t^* - β̃_{t-1}, -k, +k)
    
    This is equivalent to:
        β̃_t = β_t^*           if |β_t^* - β̃_{t-1}| ≤ k
        β̃_t = β̃_{t-1} ± k    otherwise (sign matches direction)
    
    Properties:
    - β̃_t converges to β_t^* when β_t^* is stable (monotone tracking)
    - Maximum speed of convergence is k per period
    - When k = ∞, β̃_t = β_t^* (no smoothing)
    - When k = 0, β̃_t = β̃_0 (constant beta, fully rigid)
    """
    beta_star = np.asarray(beta_unconstrained, dtype=np.float64)
    T = len(beta_star)
    beta_smooth = np.empty(T, dtype=np.float64)
    
    beta_smooth[0] = beta_star[0] if beta_init is None else beta_init
    
    for t in range(1, T):
        delta = beta_star[t] - beta_smooth[t - 1]
        clamped_delta = np.clip(delta, -max_delta_k, max_delta_k)
        beta_smooth[t] = beta_smooth[t - 1] + clamped_delta
    
    return beta_smooth


# =============================================================================
# SANDWICH MLE STANDARD ERRORS
# =============================================================================

def _numerical_gradient(nll_fn, params, data_args, eps=1e-5):
    """
    Compute per-observation score vectors via central finite differences.
    
    Returns g_t = ∂ℓ_t/∂θ for each observation t, where ℓ_t is the
    per-observation log-likelihood contribution.
    
    Parameters
    ----------
    nll_fn : callable
        Function(params, *data_args) -> per-observation NLL contributions
        (array of shape (n,)), NOT the sum.
    params : array-like, shape (p,)
        Parameter vector at which to evaluate gradients.
    data_args : tuple
        Additional arguments passed to nll_fn.
    eps : float
        Step size for finite differences.
    
    Returns
    -------
    scores : ndarray, shape (n, p)
        Score matrix. Row t is the gradient of ℓ_t w.r.t. θ.
    """
    params = np.asarray(params, dtype=np.float64)
    p = len(params)
    base = nll_fn(params, *data_args)
    n = len(base)
    scores = np.zeros((n, p))
    
    for j in range(p):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[j] += eps
        params_minus[j] -= eps
        
        f_plus = nll_fn(params_plus, *data_args)
        f_minus = nll_fn(params_minus, *data_args)
        
        # Score = -∂(nll_t)/∂θ_j  (negative because nll = -loglik)
        scores[:, j] = -(f_plus - f_minus) / (2 * eps)
    
    return scores


def _numerical_hessian(nll_fn_total, params, data_args, eps=1e-5):
    """
    Compute the Hessian of the total NLL via central finite differences.
    
    Returns H = ∂²NLL/∂θ∂θ' evaluated at params.
    
    Parameters
    ----------
    nll_fn_total : callable
        Function(params, *data_args) -> scalar total NLL.
    params : array-like, shape (p,)
    data_args : tuple
    eps : float
    
    Returns
    -------
    H : ndarray, shape (p, p)
        Hessian matrix (of NLL, so positive definite at a minimum).
    """
    params = np.asarray(params, dtype=np.float64)
    p = len(params)
    H = np.zeros((p, p))
    f0 = nll_fn_total(params, *data_args)
    
    for i in range(p):
        for j in range(i, p):
            params_pp = params.copy()
            params_mm = params.copy()
            params_pm = params.copy()
            params_mp = params.copy()
            
            params_pp[i] += eps; params_pp[j] += eps
            params_mm[i] -= eps; params_mm[j] -= eps
            params_pm[i] += eps; params_pm[j] -= eps
            params_mp[i] -= eps; params_mp[j] += eps
            
            H[i, j] = (nll_fn_total(params_pp, *data_args)
                        - nll_fn_total(params_pm, *data_args)
                        - nll_fn_total(params_mp, *data_args)
                        + nll_fn_total(params_mm, *data_args)) / (4 * eps * eps)
            H[j, i] = H[i, j]
    
    return H


def sandwich_standard_errors(nll_per_obs_fn, nll_total_fn, params_hat, data_args,
                              n_lags=None, eps=1e-5, param_names=None):
    """
    Compute sandwich (Huber-White) standard errors for MLE with dependent data.
    
    The sandwich estimator for the asymptotic covariance of θ̂ is:
    
        V = H^{-1} · S · H^{-1}
    
    where:
        H = (1/n) ∂²NLL/∂θ∂θ'              (observed information matrix)
        S = (1/n) Σ_{|h|≤L} w(h) Γ(h)      (HAC long-run variance of scores)
        Γ(h) = Σ_t g_t · g_{t+h}'          (autocovariance of scores at lag h)
        w(h) = 1 - |h|/(L+1)               (Bartlett kernel)
        g_t = ∂ℓ_t/∂θ                      (per-observation score vector)
    
    Parameters
    ----------
    nll_per_obs_fn : callable
        f(params, *data_args) -> array of shape (n,) with per-observation
        NLL contributions.
    nll_total_fn : callable
        f(params, *data_args) -> scalar total NLL.
    params_hat : array-like, shape (p,)
        MLE estimates.
    data_args : tuple
        Data arguments passed to both functions.
    n_lags : int, optional
        Newey-West truncation lag. If None, uses optimal: floor(4*(n/100)^(2/9)).
    eps : float
        Step size for numerical differentiation.
    param_names : list of str, optional
        Names for parameters (for display).
    
    Returns
    -------
    results : dict
        'sandwich_se'       : robust standard errors (array, shape (p,))
        'sandwich_cov'      : full sandwich covariance matrix (p × p)
        'hessian_se'        : naive (Hessian-only) standard errors
        'hessian_cov'       : naive covariance H^{-1}
        'hac_score_cov'     : S matrix (HAC covariance of scores)
        'n_lags'            : number of lags used
        't_statistics'      : θ̂ / se (array, shape (p,))
        'p_values'          : two-sided p-values
        'param_names'       : parameter names
        'param_table'       : formatted results table (DataFrame)
    """
    params_hat = np.asarray(params_hat, dtype=np.float64)
    p = len(params_hat)
    
    # 1. Compute per-observation scores via numerical gradient
    scores = _numerical_gradient(nll_per_obs_fn, params_hat, data_args, eps=eps)
    n = scores.shape[0]
    
    # 2. Compute Hessian of total NLL
    H = _numerical_hessian(nll_total_fn, params_hat, data_args, eps=eps)
    
    # 3. Invert Hessian (observed information)
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        # Use pseudo-inverse if Hessian is singular
        H_inv = np.linalg.pinv(H)
        print("WARNING: Hessian is singular; using pseudo-inverse.")
    
    # 4. Naive (Hessian-only) standard errors: V_naive = H^{-1}
    hessian_cov = H_inv
    hessian_se = np.sqrt(np.abs(np.diag(hessian_cov)))
    
    # 5. HAC covariance of scores (Newey-West with Bartlett kernel)
    if n_lags is None:
        n_lags = int(np.floor(4 * (n / 100) ** (2/9)))
    n_lags = max(n_lags, 1)
    
    # Demean scores (they should be zero-mean at MLE, but numerical noise)
    g = scores - scores.mean(axis=0)
    
    # Lag-0 autocovariance: Γ(0) = Σ_t g_t g_t'
    S = g.T @ g  # (p × p)
    
    # Add lagged autocovariances with Bartlett kernel
    for h in range(1, n_lags + 1):
        weight = 1 - h / (n_lags + 1)
        Gamma_h = g[h:].T @ g[:-h]  # (p × p)
        S += weight * (Gamma_h + Gamma_h.T)
    
    # 6. Sandwich covariance: V = H^{-1} S H^{-1}
    sandwich_cov = H_inv @ S @ H_inv
    
    # Ensure positive diagonal elements
    diag_vals = np.diag(sandwich_cov)
    if np.any(diag_vals < 0):
        print("WARNING: Some diagonal elements of sandwich covariance are negative.")
        print("         This may indicate numerical issues. Using absolute values.")
    
    sandwich_se = np.sqrt(np.abs(diag_vals))
    
    # 7. t-statistics and p-values (using normal distribution for MLE)
    t_stats = params_hat / sandwich_se
    p_values = 2 * (1 - stats.norm.cdf(np.abs(t_stats)))
    
    # 8. Build parameter names
    if param_names is None:
        param_names = [f'param_{i}' for i in range(p)]
    
    # 9. Build results table
    table_data = {
        'Parameter': param_names,
        'Estimate': params_hat,
        'Naive SE': hessian_se,
        'Sandwich SE': sandwich_se,
        'SE Ratio': sandwich_se / np.where(hessian_se > 0, hessian_se, np.nan),
        't-stat': t_stats,
        'p-value': p_values,
        'Sig.': ['***' if pv < 0.001 else '**' if pv < 0.01 else '*' if pv < 0.05
                 else '.' if pv < 0.1 else '' for pv in p_values]
    }
    param_table = pd.DataFrame(table_data)
    
    return {
        'sandwich_se': sandwich_se,
        'sandwich_cov': sandwich_cov,
        'hessian_se': hessian_se,
        'hessian_cov': hessian_cov,
        'hac_score_cov': S / n,
        'n_lags': n_lags,
        't_statistics': t_stats,
        'p_values': p_values,
        'param_names': param_names,
        'param_table': param_table,
        'scores': scores,
        'n_obs': n
    }


# =============================================================================
# ENHANCED MODEL CLASS
# =============================================================================

class EnhancedDynamicBetaModel:
    """
    MMDA Dynamic Beta Model v2.0 with AR-Smoothed Betas and Sandwich SEs.
    
    Improvements over v1.1:
    1. AR smoothing constraint: |β_t - β_{t-1}| ≤ k prevents overly volatile betas
    2. Sandwich MLE standard errors for ALL parameters (including nonlinear)
    3. Both naive Hessian SEs and HAC-robust sandwich SEs reported
    
    Usage
    -----
    >>> model = EnhancedDynamicBetaModel(max_beta_change=0.02)
    >>> model.load_data('bankratemma.csv')
    >>> results = model.estimate()
    >>> se_results = model.compute_sandwich_se()
    >>> model.print_results()
    """
    
    def __init__(self, max_beta_change=0.02, volatility_window=24, min_periods=12):
        """
        Parameters
        ----------
        max_beta_change : float
            Maximum allowable absolute change in beta per period.
            Default 0.02 = 2 pp/month. Set to np.inf to disable smoothing.
        volatility_window : int
            Rolling window for volatility calculation (months).
        min_periods : int
            Minimum observations for rolling volatility.
        """
        self.max_beta_change = max_beta_change
        self.volatility_window = volatility_window
        self.min_periods = min_periods
        
        # Storage
        self.data = None
        self.result = None
        self.params = None
        self.beta_unconstrained = None
        self.beta_smoothed = None
        self.predictions = None
        self.residuals = None
        self.se_results = None
    
    def load_data(self, filepath, start_date='2017-01-01', end_date='2025-03-31'):
        """Load and prepare the dataset."""
        df = pd.read_csv(filepath)
        df['EOM_Dt'] = pd.to_datetime(df['EOM_Dt'])
        
        # Clean term spread columns
        term_spread_cols = ['3M_1M_SPRD', '6M_3M_SPRD', '1Y_3M_SPRD', 
                           '2Y_3M_SPRD', '3Y_3M_SPRD', '5Y_3M_SPRD', '10Y_3M_SPRD']
        for col in term_spread_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace('(', '-').str.replace(')', '')
                df[col] = df[col].str.strip().str.replace(r'[^0-9.-]', '', regex=True)
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Filter to analysis period
        mask = (df['EOM_Dt'] >= start_date) & (df['EOM_Dt'] <= end_date)
        df = df.loc[mask].copy().reset_index(drop=True)
        
        # Calculate volatility measures
        df['FEDL01_change'] = df['FEDL01'].diff()
        df['vol_24m'] = df['FEDL01_change'].rolling(
            window=self.volatility_window, min_periods=self.min_periods
        ).std()
        
        first_valid_idx = df['vol_24m'].first_valid_index()
        if first_valid_idx is not None:
            df['vol_24m'] = df['vol_24m'].fillna(df.loc[first_valid_idx, 'vol_24m'])
        
        df['vol_star'] = df['vol_24m'].expanding().mean()
        df['vol_ratio'] = df['vol_24m'] / df['vol_star']
        
        self.data = df
        
        # Extract arrays
        self.Y = df['ILMDHYLD'].values
        self.FEDL01 = df['FEDL01'].values
        self.FHLK3MSPRD = df['FHLK3MSPRD'].values
        self.vol_ratio = df['vol_ratio'].values
        self.rate_change = df['FEDL01_change'].fillna(0).values
        
        print(f"Loaded {len(df)} observations from {df['EOM_Dt'].min().date()} to {df['EOM_Dt'].max().date()}")
        print(f"Fed Funds range: {self.FEDL01.min():.2f}% - {self.FEDL01.max():.2f}%")
        print(f"MMDA rate range: {self.Y.min():.2f}% - {self.Y.max():.2f}%")
        print(f"AR smoothing constraint: max Δβ = {self.max_beta_change:.4f} per period")
        
        return df
    
    # -------------------------------------------------------------------------
    # Per-observation NLL (for sandwich SE computation)
    # -------------------------------------------------------------------------
    
    def _nll_per_obs(self, params, Y, FEDL01, FHLK3MSPRD, vol_ratio, rate_change):
        """
        Per-observation negative log-likelihood contributions.
        
        Returns array of shape (n,) where element t is:
            nll_t = 0.5 * log(2π σ²) + (y_t - ŷ_t)² / (2σ²)
        
        NOTE: σ² is estimated as the sample variance of residuals (concentrated
        out of the likelihood). This means we treat σ² as a function of θ for
        the purposes of per-observation contributions.
        """
        alpha, k, m, beta_min, beta_max, gamma_fhlb, lambda_up, lambda_down = params
        
        beta_unc = asymmetric_volatility_beta(
            FEDL01, k, m, beta_min, beta_max, vol_ratio,
            lambda_up, lambda_down, rate_change
        )
        beta = ar_smoothed_beta(beta_unc, self.max_beta_change)
        
        pred = alpha + beta * FEDL01 + gamma_fhlb * FHLK3MSPRD
        residuals = Y - pred
        sigma_sq = np.var(residuals)
        
        if sigma_sq <= 0 or not np.isfinite(sigma_sq):
            return np.full(len(Y), 1e10 / len(Y))
        
        nll_t = 0.5 * np.log(2 * np.pi * sigma_sq) + residuals**2 / (2 * sigma_sq)
        return nll_t
    
    def _nll_total(self, params, Y, FEDL01, FHLK3MSPRD, vol_ratio, rate_change):
        """Total negative log-likelihood (scalar)."""
        return np.sum(self._nll_per_obs(params, Y, FEDL01, FHLK3MSPRD, vol_ratio, rate_change))
    
    # -------------------------------------------------------------------------
    # Estimation
    # -------------------------------------------------------------------------
    
    def estimate(self, n_starts=5):
        """
        Estimate the enhanced model with AR-smoothed betas.
        
        Parameters
        ----------
        n_starts : int
            Number of random starting points for global optimization.
        
        Returns
        -------
        result : scipy.optimize.OptimizeResult
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        Y = self.Y
        FEDL01 = self.FEDL01
        FHLK3MSPRD = self.FHLK3MSPRD
        vol_ratio = self.vol_ratio
        rate_change = self.rate_change
        
        def nll(params):
            return self._nll_total(params, Y, FEDL01, FHLK3MSPRD, vol_ratio, rate_change)
        
        # Bounds: [alpha, k, m, beta_min, beta_max, gamma_fhlb, lambda_up, lambda_down]
        bounds = [
            (-1, 1),        # alpha
            (0.01, 5),      # k (steepness)
            (0.5, 5),       # m (inflection)
            (0.30, 0.50),   # beta_min
            (0.55, 0.80),   # beta_max
            (-3, 3),        # gamma_fhlb
            (0, 1),         # lambda_up
            (0, 1),         # lambda_down
        ]
        
        starts = [
            [0.0, 0.5, 2.5, 0.40, 0.70, 1.0, 0.25, 0.20],
            [0.1, 0.8, 3.0, 0.35, 0.65, 0.5, 0.30, 0.15],
            [-0.1, 0.4, 2.0, 0.42, 0.68, 1.5, 0.20, 0.25],
            [0.0, 1.0, 3.5, 0.38, 0.72, 0.8, 0.15, 0.10],
            [0.05, 0.6, 2.8, 0.36, 0.66, 1.2, 0.28, 0.22],
        ]
        
        best_result = None
        best_nll = np.inf
        
        for i, start in enumerate(starts[:n_starts]):
            try:
                result = minimize(
                    nll, start, bounds=bounds, method='L-BFGS-B',
                    options={'maxiter': 2000, 'ftol': 1e-9}
                )
                if result.fun < best_nll:
                    best_nll = result.fun
                    best_result = result
            except Exception as e:
                print(f"  Start {i+1} failed: {e}")
        
        if best_result is None:
            raise RuntimeError("All optimization attempts failed.")
        
        self.result = best_result
        params = best_result.x
        self.params = {
            'alpha': params[0],
            'k': params[1],
            'm': params[2],
            'beta_min': params[3],
            'beta_max': params[4],
            'gamma_fhlb': params[5],
            'lambda_up': params[6],
            'lambda_down': params[7],
        }
        
        # Compute betas
        self.beta_unconstrained = asymmetric_volatility_beta(
            FEDL01, params[1], params[2], params[3], params[4],
            vol_ratio, params[6], params[7], rate_change
        )
        self.beta_smoothed = ar_smoothed_beta(
            self.beta_unconstrained, self.max_beta_change
        )
        
        # Predictions and residuals
        self.predictions = params[0] + self.beta_smoothed * FEDL01 + params[5] * FHLK3MSPRD
        self.residuals = Y - self.predictions
        
        return best_result
    
    # -------------------------------------------------------------------------
    # Sandwich Standard Errors
    # -------------------------------------------------------------------------
    
    def compute_sandwich_se(self, n_lags=None, eps=1e-5):
        """
        Compute sandwich (robust) standard errors for ALL model parameters.
        
        Uses the Huber-White sandwich estimator with Newey-West HAC weighting
        for the outer product of scores, properly accounting for serial
        dependence in the time series.
        
        Parameters
        ----------
        n_lags : int, optional
            Number of Newey-West lags. If None, uses optimal lag selection.
        eps : float
            Step size for numerical differentiation.
        
        Returns
        -------
        results : dict
            Contains sandwich_se, hessian_se, sandwich_cov, param_table, etc.
        """
        if self.result is None:
            raise ValueError("Model not estimated. Call estimate() first.")
        
        param_names = ['alpha', 'k', 'm', 'beta_min', 'beta_max',
                       'gamma_fhlb', 'lambda_up', 'lambda_down']
        
        data_args = (self.Y, self.FEDL01, self.FHLK3MSPRD,
                     self.vol_ratio, self.rate_change)
        
        print("Computing sandwich standard errors...")
        print("  Step 1: Numerical score vectors (per-observation gradients)...")
        print("  Step 2: Numerical Hessian of total NLL...")
        print("  Step 3: HAC long-run variance of scores (Newey-West)...")
        print("  Step 4: Assembling sandwich covariance V = H⁻¹ S H⁻¹...")
        
        self.se_results = sandwich_standard_errors(
            nll_per_obs_fn=self._nll_per_obs,
            nll_total_fn=self._nll_total,
            params_hat=self.result.x,
            data_args=data_args,
            n_lags=n_lags,
            eps=eps,
            param_names=param_names
        )
        
        print(f"  Done. Used {self.se_results['n_lags']} Newey-West lags.")
        
        return self.se_results
    
    # -------------------------------------------------------------------------
    # Model Fit Metrics
    # -------------------------------------------------------------------------
    
    def compute_metrics(self):
        """Compute standard model fit metrics."""
        if self.residuals is None:
            raise ValueError("Model not estimated.")
        
        Y = self.Y
        pred = self.predictions
        resid = self.residuals
        n = len(Y)
        p = 8  # number of parameters
        
        ss_res = np.sum(resid**2)
        ss_tot = np.sum((Y - Y.mean())**2)
        
        r2 = 1 - ss_res / ss_tot
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        rmse = np.sqrt(np.mean(resid**2))
        mae = np.mean(np.abs(resid))
        aic = n * np.log(ss_res / n) + 2 * p
        bic = n * np.log(ss_res / n) + p * np.log(n)
        
        # Beta smoothness metrics
        beta_changes = np.abs(np.diff(self.beta_smoothed))
        beta_changes_unc = np.abs(np.diff(self.beta_unconstrained))
        
        metrics = {
            'r_squared': r2,
            'adj_r_squared': adj_r2,
            'rmse': rmse,
            'mae': mae,
            'aic': aic,
            'bic': bic,
            'n_obs': n,
            'n_params': p,
            'max_beta_change_actual': beta_changes.max(),
            'mean_beta_change_actual': beta_changes.mean(),
            'max_beta_change_unconstrained': beta_changes_unc.max(),
            'mean_beta_change_unconstrained': beta_changes_unc.mean(),
            'n_constrained_periods': np.sum(beta_changes_unc > self.max_beta_change),
            'sigma_hat': np.std(resid),
        }
        
        return metrics
    
    # -------------------------------------------------------------------------
    # Reporting
    # -------------------------------------------------------------------------
    
    def print_results(self):
        """Print comprehensive model results."""
        if self.result is None:
            raise ValueError("Model not estimated.")
        
        metrics = self.compute_metrics()
        p = self.params
        
        print("\n" + "=" * 90)
        print(" ENHANCED DYNAMIC BETA MODEL v2.0")
        print(" AR-Smoothed Betas with Sandwich Standard Errors")
        print("=" * 90)
        
        print(f"""
MODEL SPECIFICATION:
────────────────────────────────────────────────────────────────────────────────────────
  MMDA_t = α + β̃_t × FEDL01_t + γ × FHLK3MSPRD_t + ε_t

  where:
    β_t^* = β_min + (β_max - β_min) / (1 + exp(-k(r_t - m))) × (1 - λ_eff × vol_ratio_t)
    
    λ_eff = λ_up if Δr > 0, else λ_down        (asymmetric volatility dampening)
    
    β̃_t = β̃_{{t-1}} + clip(β_t^* - β̃_{{t-1}}, -{self.max_beta_change}, +{self.max_beta_change})
                                                  (AR smoothing, max Δβ = {self.max_beta_change})

ESTIMATED PARAMETERS:
────────────────────────────────────────────────────────────────────────────────────────""")
        
        if self.se_results is not None:
            # Print with standard errors
            print(f"\n  {'Parameter':<16} {'Estimate':>10} {'Naive SE':>10} {'Sandwich SE':>12} {'t-stat':>8} {'p-value':>9}  {'Sig':>4}")
            print("  " + "─" * 75)
            
            tbl = self.se_results['param_table']
            for _, row in tbl.iterrows():
                print(f"  {row['Parameter']:<16} {row['Estimate']:>10.5f} {row['Naive SE']:>10.5f} "
                      f"{row['Sandwich SE']:>12.5f} {row['t-stat']:>8.3f} {row['p-value']:>9.4f}  {row['Sig.']:>4}")
            
            print("  " + "─" * 75)
            print(f"  Newey-West lags: {self.se_results['n_lags']}")
            
            # Highlight SE ratio
            print(f"\n  SE RATIO (Sandwich / Naive):")
            print("  " + "─" * 50)
            for _, row in tbl.iterrows():
                ratio = row['SE Ratio']
                flag = " ← UNDERESTIMATED" if ratio > 1.5 else " ← OVERESTIMATED" if ratio < 0.67 else ""
                print(f"  {row['Parameter']:<16} {ratio:>8.2f}x{flag}")
            print(f"""
  NOTE: SE ratios > 1 indicate the naive Hessian SEs UNDERSTATE uncertainty.
  This is expected with serially correlated observations. The sandwich SE
  accounts for this dependence via Newey-West HAC on the score vectors.""")
        
        else:
            # Print without standard errors
            for name, val in p.items():
                print(f"  {name:<16} {val:>10.5f}")
            print("\n  (Run compute_sandwich_se() for robust standard errors)")
        
        print(f"""
MODEL FIT:
────────────────────────────────────────────────────────────────────────────────────────
  R²            = {metrics['r_squared']:.4f}
  Adjusted R²   = {metrics['adj_r_squared']:.4f}
  RMSE          = {metrics['rmse']:.4f}%
  MAE           = {metrics['mae']:.4f}%
  AIC           = {metrics['aic']:.1f}
  BIC           = {metrics['bic']:.1f}
  σ̂             = {metrics['sigma_hat']:.4f}%
  n             = {metrics['n_obs']}
  
BETA SMOOTHNESS:
────────────────────────────────────────────────────────────────────────────────────────
  Max Δβ constraint (k):   {self.max_beta_change:.4f}
  Max Δβ (unconstrained):  {metrics['max_beta_change_unconstrained']:.4f}
  Max Δβ (smoothed):       {metrics['max_beta_change_actual']:.4f}
  Mean Δβ (unconstrained): {metrics['mean_beta_change_unconstrained']:.4f}
  Mean Δβ (smoothed):      {metrics['mean_beta_change_actual']:.4f}
  Periods constrained:     {metrics['n_constrained_periods']} / {metrics['n_obs'] - 1}

INTERPRETATION OF AR-SMOOTHING:
────────────────────────────────────────────────────────────────────────────────────────
  The unconstrained beta can jump by up to {metrics['max_beta_change_unconstrained']*100:.1f} pp in a single month.
  With the AR constraint (k = {self.max_beta_change}), the max change is capped at {self.max_beta_change*100:.1f} pp/month.
  The smoothed beta still converges to the equilibrium S-curve value but does so
  gradually, preventing implausible jumps in deposit repricing behavior.
""")
        
        # S-curve beta at key rate levels
        print("EQUILIBRIUM BETA BY RATE LEVEL:")
        print("────────────────────────────────────────────────────────────────────────────────────────")
        r_range = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        beta_range = logistic_beta(r_range, p['k'], p['m'], p['beta_min'], p['beta_max'])
        
        print(f"\n  {'Fed Funds':>10}  {'Eq. Beta':>10}")
        print("  " + "─" * 30)
        for r, b in zip(r_range, beta_range):
            bar = "█" * int(b * 50)
            print(f"    {r:>4.0f}%      {b*100:>5.1f}%  {bar}")
    
    def compare_with_unconstrained(self):
        """
        Compare AR-smoothed vs unconstrained beta visually and numerically.
        """
        if self.beta_smoothed is None:
            raise ValueError("Model not estimated.")
        
        # Also compute unconstrained model predictions for comparison
        pred_unc = (self.params['alpha'] + self.beta_unconstrained * self.FEDL01
                    + self.params['gamma_fhlb'] * self.FHLK3MSPRD)
        resid_unc = self.Y - pred_unc
        
        ss_res_unc = np.sum(resid_unc**2)
        ss_res_smooth = np.sum(self.residuals**2)
        ss_tot = np.sum((self.Y - self.Y.mean())**2)
        
        r2_unc = 1 - ss_res_unc / ss_tot
        r2_smooth = 1 - ss_res_smooth / ss_tot
        rmse_unc = np.sqrt(np.mean(resid_unc**2))
        rmse_smooth = np.sqrt(np.mean(self.residuals**2))
        
        # Beta change statistics
        changes_unc = np.abs(np.diff(self.beta_unconstrained))
        changes_smooth = np.abs(np.diff(self.beta_smoothed))
        
        print("\n" + "=" * 80)
        print(" COMPARISON: Unconstrained vs AR-Smoothed Beta")
        print("=" * 80)
        
        print(f"""
                          {'Unconstrained':>16} {'AR-Smoothed':>16} {'Difference':>12}
  ────────────────────────────────────────────────────────────────────────────
  R²                     {r2_unc:>16.4f} {r2_smooth:>16.4f} {r2_smooth - r2_unc:>+12.4f}
  RMSE                   {rmse_unc:>15.4f}% {rmse_smooth:>15.4f}% {rmse_smooth - rmse_unc:>+11.4f}%
  Max |Δβ|               {changes_unc.max():>16.4f} {changes_smooth.max():>16.4f} {changes_smooth.max() - changes_unc.max():>+12.4f}
  Mean |Δβ|              {changes_unc.mean():>16.4f} {changes_smooth.mean():>16.4f} {changes_smooth.mean() - changes_unc.mean():>+12.4f}
  Std |Δβ|               {changes_unc.std():>16.4f} {changes_smooth.std():>16.4f} {changes_smooth.std() - changes_unc.std():>+12.4f}
  95th pctl |Δβ|         {np.percentile(changes_unc, 95):>16.4f} {np.percentile(changes_smooth, 95):>16.4f}
  Periods >k             {np.sum(changes_unc > self.max_beta_change):>16d} {np.sum(changes_smooth > self.max_beta_change):>16d}
  ────────────────────────────────────────────────────────────────────────────

  The AR constraint reduces beta volatility while maintaining good model fit.
  R² drops by {abs(r2_smooth - r2_unc):.4f} — a small cost for much smoother betas.
""")
        
        return {
            'r2_unconstrained': r2_unc,
            'r2_smoothed': r2_smooth,
            'rmse_unconstrained': rmse_unc,
            'rmse_smoothed': rmse_smooth,
        }
    
    def sensitivity_analysis_k(self, k_values=None):
        """
        Sensitivity analysis: how does model fit change with different
        smoothing constraints k?
        
        Parameters
        ----------
        k_values : list of float, optional
            Values of max_beta_change to test. If None, uses default range.
        
        Returns
        -------
        results : pd.DataFrame
            Table of metrics for each k value.
        """
        if self.result is None:
            raise ValueError("Model not estimated.")
        
        if k_values is None:
            k_values = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.05, 0.10, np.inf]
        
        params = self.result.x
        Y = self.Y
        FEDL01 = self.FEDL01
        FHLK3MSPRD = self.FHLK3MSPRD
        
        rows = []
        for kv in k_values:
            beta_s = ar_smoothed_beta(self.beta_unconstrained, kv)
            pred = params[0] + beta_s * FEDL01 + params[5] * FHLK3MSPRD
            resid = Y - pred
            
            ss_res = np.sum(resid**2)
            ss_tot = np.sum((Y - Y.mean())**2)
            r2 = 1 - ss_res / ss_tot
            rmse = np.sqrt(np.mean(resid**2))
            changes = np.abs(np.diff(beta_s))
            
            rows.append({
                'k (max Δβ)': kv if np.isfinite(kv) else '∞',
                'R²': r2,
                'RMSE': rmse,
                'Max |Δβ|': changes.max(),
                'Mean |Δβ|': changes.mean(),
                'Periods constrained': np.sum(np.abs(np.diff(self.beta_unconstrained)) > kv)
            })
        
        df_results = pd.DataFrame(rows)
        
        print("\n" + "=" * 90)
        print(" SENSITIVITY ANALYSIS: Beta Smoothing Constraint k")
        print("=" * 90)
        print(f"\n{df_results.to_string(index=False)}\n")
        print("  NOTE: k = ∞ recovers the original (unconstrained) model.\n"
              "  Choose k to balance model fit (R²) against beta stability (Max |Δβ|).\n"
              "  Typical recommendation: k = 0.02 (2 pp/month) based on economic reasonableness.")
        
        return df_results
    
    def plot_results(self, save_path=None):
        """Generate comprehensive visualization of model results."""
        import matplotlib.pyplot as plt
        
        if self.beta_smoothed is None:
            raise ValueError("Model not estimated.")
        
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        dates = self.data['EOM_Dt']
        p = self.params
        
        # Panel A: S-Curve Beta Function
        ax = axes[0, 0]
        r_plot = np.linspace(0, 10, 100)
        beta_plot = logistic_beta(r_plot, p['k'], p['m'], p['beta_min'], p['beta_max']) * 100
        ax.plot(r_plot, beta_plot, 'b-', linewidth=3, label='Equilibrium β(r)')
        ax.axhline(y=p['beta_min'] * 100, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=p['beta_max'] * 100, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=p['m'], color='red', linestyle=':', alpha=0.7, label=f"m = {p['m']:.1f}%")
        ax.set_xlabel('Fed Funds Rate (%)')
        ax.set_ylabel('Equilibrium Beta (%)')
        ax.set_title('A: S-Curve Equilibrium Beta', fontweight='bold')
        ax.legend(loc='lower right')
        
        # Panel B: Unconstrained vs Smoothed Beta
        ax = axes[0, 1]
        ax.plot(dates, self.beta_unconstrained * 100, 'r-', alpha=0.5, linewidth=1,
                label='Unconstrained β*')
        ax.plot(dates, self.beta_smoothed * 100, 'b-', linewidth=2.5,
                label=f'AR-Smoothed β̃ (k={self.max_beta_change})')
        ax.set_xlabel('Date')
        ax.set_ylabel('Beta (%)')
        ax.set_title('B: Beta Smoothing Effect', fontweight='bold')
        ax.legend()
        
        # Panel C: Model Fit
        ax = axes[0, 2]
        ax.plot(dates, self.Y, 'ko', markersize=4, alpha=0.6, label='Actual MMDA')
        ax.plot(dates, self.predictions, 'b-', linewidth=2, label='Model Prediction')
        metrics = self.compute_metrics()
        ax.set_xlabel('Date')
        ax.set_ylabel('MMDA Rate (%)')
        ax.set_title(f"C: Model Fit (R² = {metrics['r_squared']:.4f})", fontweight='bold')
        ax.legend()
        
        # Panel D: Residuals
        ax = axes[1, 0]
        ax.bar(dates, self.residuals, width=20, alpha=0.7, color='steelblue')
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.set_xlabel('Date')
        ax.set_ylabel('Residual (%)')
        ax.set_title(f"D: Residuals (RMSE = {metrics['rmse']:.4f}%)", fontweight='bold')
        
        # Panel E: Period-to-period beta changes
        ax = axes[1, 1]
        changes_unc = np.diff(self.beta_unconstrained) * 100
        changes_smooth = np.diff(self.beta_smoothed) * 100
        ax.plot(dates[1:], changes_unc, 'r-', alpha=0.4, linewidth=1, label='Unconstrained Δβ')
        ax.plot(dates[1:], changes_smooth, 'b-', linewidth=2, label='Smoothed Δβ')
        ax.axhline(y=self.max_beta_change * 100, color='green', linestyle='--', alpha=0.7, label=f'±k = ±{self.max_beta_change*100:.1f} pp')
        ax.axhline(y=-self.max_beta_change * 100, color='green', linestyle='--', alpha=0.7)
        ax.set_xlabel('Date')
        ax.set_ylabel('Δβ (pp/month)')
        ax.set_title('E: Period-to-Period Beta Changes', fontweight='bold')
        ax.legend(fontsize=8)
        
        # Panel F: SE comparison (if available)
        ax = axes[1, 2]
        if self.se_results is not None:
            tbl = self.se_results['param_table']
            x_pos = np.arange(len(tbl))
            width = 0.35
            ax.bar(x_pos - width/2, tbl['Naive SE'].values, width, label='Naive (Hessian)', alpha=0.7, color='skyblue')
            ax.bar(x_pos + width/2, tbl['Sandwich SE'].values, width, label='Sandwich (HAC)', alpha=0.7, color='coral')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(tbl['Parameter'].values, rotation=45, ha='right')
            ax.set_ylabel('Standard Error')
            ax.set_title('F: Naive vs Sandwich SE', fontweight='bold')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'Run compute_sandwich_se()\nfor SE comparison',
                    transform=ax.transAxes, ha='center', va='center', fontsize=12)
            ax.set_title('F: Standard Errors (not yet computed)', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"Figure saved: {save_path}")
        
        plt.close()
        return fig


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    
    print("=" * 90)
    print(" MMDA Dynamic Beta Model v2.0")
    print(" Enhancements: AR-Smoothed Betas + Sandwich MLE Standard Errors")
    print("=" * 90)
    
    # -------------------------------------------------------------------------
    # 1. Estimate model with default smoothing constraint
    # -------------------------------------------------------------------------
    model = EnhancedDynamicBetaModel(max_beta_change=0.02)
    model.load_data('bankratemma.csv')
    
    print("\n" + "-" * 90)
    print("Estimating enhanced model...")
    print("-" * 90)
    result = model.estimate(n_starts=5)
    print(f"Optimization converged: {result.success}")
    print(f"Final NLL: {result.fun:.4f}")
    
    # -------------------------------------------------------------------------
    # 2. Compute sandwich standard errors
    # -------------------------------------------------------------------------
    print("\n" + "-" * 90)
    print("Computing robust (sandwich) standard errors...")
    print("-" * 90)
    se_results = model.compute_sandwich_se()
    
    # -------------------------------------------------------------------------
    # 3. Print full results
    # -------------------------------------------------------------------------
    model.print_results()
    
    # -------------------------------------------------------------------------
    # 4. Compare smoothed vs unconstrained
    # -------------------------------------------------------------------------
    model.compare_with_unconstrained()
    
    # -------------------------------------------------------------------------
    # 5. Sensitivity analysis on smoothing parameter k
    # -------------------------------------------------------------------------
    model.sensitivity_analysis_k()
    
    # -------------------------------------------------------------------------
    # 6. Generate visualization
    # -------------------------------------------------------------------------
    import os
    os.makedirs('outputs/figures', exist_ok=True)
    model.plot_results(save_path='outputs/figures/enhanced_model_v2.png')
    
    # -------------------------------------------------------------------------
    # 7. Methodology notes
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print(" METHODOLOGY NOTES")
    print("=" * 90)
    print(f"""
1. AR-SMOOTHED BETA (Autoregressive Structure)
   ─────────────────────────────────────────────────────────────────────────────────
   The comment suggested adding an AR structure such that |β(t) - β(t-1)| ≤ k.
   
   Implementation: β̃_t = β̃_{{t-1}} + clip(β_t^* - β̃_{{t-1}}, -k, +k)
   
   This is a projection-based AR(1) smoother. The unconstrained (S-curve + volatility)
   beta β_t^* serves as the "target", and the smoothed beta β̃_t tracks it subject
   to the maximum-change constraint k.
   
   Economic rationale: Banks cannot instantaneously change their deposit repricing
   behavior. Even if the equilibrium beta shifts rapidly (e.g., during a 75bp Fed
   hike), actual pricing behavior changes more gradually due to:
     a) Administrative pricing committees with monthly/quarterly review cycles
     b) Competitive dynamics requiring observation of peer pricing
     c) Customer relationship management considerations
   
   The tolerance k is user-supplied and can be calibrated to institutional pricing
   cycles. Default k = 0.02 (2 pp/month) is based on observed pricing behavior
   during the 2022-2023 hiking cycle.

2. SANDWICH MLE STANDARD ERRORS
   ─────────────────────────────────────────────────────────────────────────────────
   The comment correctly identified that the original MLE SEs were incorrect because
   observations are not independent across time.
   
   The original code used either:
   (a) No SEs for nonlinear parameters, or
   (b) Newey-West HAC on a linearized approximation (only for α, γ_FHLB)
   
   Both approaches are incomplete. The correct asymptotic covariance for MLE with
   dependent observations is the Huber-White sandwich estimator:
   
       V_sandwich = H⁻¹ · S · H⁻¹
   
   where:
   - H = observed information matrix (Hessian of NLL)
   - S = long-run variance of the score vector, estimated with Newey-West HAC
   - g_t = ∂ℓ_t/∂θ = per-observation score (gradient of individual log-likelihood)
   
   Under correct specification AND independence, H⁻¹ = S⁻¹ and the sandwich
   collapses to the standard Hessian-based variance V = H⁻¹.
   
   With serial correlation, S ≠ H (the scores are autocorrelated), so:
   - Naive V = H⁻¹ underestimates uncertainty (SEs too small)
   - Sandwich V = H⁻¹·S·H⁻¹ properly inflates for dependence
   
   The SE ratio (sandwich/naive) indicates how much the naive SEs understate
   uncertainty. Values > 1 are expected and common for time series MLE.
   
   Reference: White, H. (1982). "Maximum likelihood estimation of misspecified 
   models." Econometrica, 50(1), 1-25.
""")
