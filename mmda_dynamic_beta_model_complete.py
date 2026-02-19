"""
Money Market Deposit Account Dynamic Beta Repricing Model
Complete Implementation with Analysis and Visualizations

Model Owner: [Author Name]
Department: Asset Liability Management / Treasury
Date: October 1, 2025
Model Version: 1.1

This code provides complete implementation of the volatility-adjusted dynamic beta model
for MMDA repricing, including all analysis, testing, and visualizations required
for model development and validation.

RECOMMENDED MODEL SPECIFICATION (v1.1):
=======================================
The production model uses 8 parameters (asymmetric volatility WITHOUT term spread):
    MMDA_t = α + β_t × FEDL01_t + γ × FHLK3MSPRD_t + ε_t

where β_t = β_min + (β_max - β_min) × logistic(k, m, FEDL01) × volatility_adjustment

Parameters: α, k, m, β_min, β_max, γ_FHLB, λ_up, λ_down

NOTE ON TERM SPREAD REMOVAL:
The 1Y-3M term spread variable was removed from the model specification (v1.0 → v1.1)
due to spurious correlation. During the 2022-2023 hiking cycle, the Fed's aggressive
rate increases caused both: (1) MMDA rates to rise, and (2) the yield curve to invert.
This created a mechanical negative correlation (r = -0.925) between term spread and
MMDA rates that is not causal. The λ_up volatility dampening parameter already captures
banks' tendency to lag rate hikes, making term spread redundant and misleading.

Use estimate_asymmetric_volatility_revised() for the production model.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss, coint
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.stats.diagnostic import acorr_breusch_godfrey, het_white
from statsmodels.stats.stattools import jarque_bera
from statsmodels.regression.linear_model import OLSResults
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Set style for professional visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

class MMDADynamicBetaModel:
    """
    Complete implementation of the MMDA Dynamic Beta Repricing Model
    
    This class provides comprehensive functionality for model development,
    validation, and deployment including data preprocessing, model estimation,
    validation testing, and visualization capabilities.
    
    RECOMMENDED METHOD: estimate_asymmetric_volatility_revised()
    This 8-parameter model excludes term spread which showed spurious correlation.
    
    Key Model Methods:
    - estimate_asymmetric_volatility_revised(): PRODUCTION model (8 params, no term spread)
    - estimate_asymmetric_volatility(): Legacy model (9 params, with term spread)
    - estimate_volatility_adjusted(): Symmetric volatility model
    - estimate_enhanced_logistic(): Level-dependent beta only
    """
    
    def __init__(self, model_version="1.0"):
        self.model_version = model_version
        self.models = {}
        self.data = None
        self.results = {}
        self.validation_results = {}
        
        # Model configuration
        self.config = {
            'volatility_window': 24,
            'min_periods': 12,
            'optimization_method': 'L-BFGS-B',
            'tolerance': 1e-9,
            'max_iterations': 1000
        }
        
    def load_and_prepare_data(self, filepath, start_date='2017-01-01', end_date='2025-03-31'):
        """
        Load and prepare dataset with comprehensive validation and preprocessing
        
        Parameters:
        -----------
        filepath : str
            Path to the CSV data file
        start_date : str
            Start date for analysis period
        end_date : str
            End date for analysis period
            
        Returns:
        --------
        pandas.DataFrame
            Processed dataset ready for modeling
        """
        
        try:
            print("Loading data from Bloomberg/FRED sources...")
            df = pd.read_csv(filepath)
            df['EOM_Dt'] = pd.to_datetime(df['EOM_Dt'])
            print(f"Successfully loaded {len(df)} rows of data")
            
            # Clean term spread columns systematically
            term_spread_cols = ['3M_1M_SPRD', '6M_3M_SPRD', '1Y_3M_SPRD', 
                               '2Y_3M_SPRD', '3Y_3M_SPRD', '5Y_3M_SPRD', '10Y_3M_SPRD']
            
            for col in term_spread_cols:
                if col in df.columns:
                    # Handle parenthetical negative values and clean non-numeric characters
                    df[col] = df[col].astype(str).str.replace('(', '-').str.replace(')', '')
                    df[col] = df[col].str.strip().str.replace(r'[^0-9.-]', '', regex=True)
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    print(f"Cleaned {col}: {df[col].count()} valid values")
            
            # Filter to analysis period
            mask = (df['EOM_Dt'] >= start_date) & (df['EOM_Dt'] <= end_date)
            df = df.loc[mask].copy()
            print(f"Filtered to analysis period {start_date} to {end_date}: {len(df)} observations")
            
            # Calculate volatility measures with extended window for stability
            df['FEDL01_change'] = df['FEDL01'].diff()
            df['vol_24m'] = df['FEDL01_change'].rolling(
                window=self.config['volatility_window'], 
                min_periods=self.config['min_periods']
            ).std()
            
            # Fill initial NaNs with first valid value
            first_valid_idx = df['vol_24m'].first_valid_index()
            if first_valid_idx is not None:
                first_valid_value = df.loc[first_valid_idx, 'vol_24m']
                df['vol_24m'] = df['vol_24m'].fillna(first_valid_value)
            
            # Calculate long-run volatility statistics
            df['vol_star'] = df['vol_24m'].expanding().mean()
            df['vol_ratio'] = df['vol_24m'] / df['vol_star']
            
            # Data quality validation
            self._validate_data_quality(df)
            
            self.data = df
            return df
            
        except Exception as e:
            print(f"Error loading or preparing data: {e}")
            return None
    
    def _validate_data_quality(self, df):
        """Comprehensive data quality validation and reporting"""
        
        print("\nData Quality Assessment:")
        print("-" * 40)
        
        # Check for required columns
        required_cols = ['ILMDHYLD', 'FEDL01', 'FHLK3MSPRD', '1Y_3M_SPRD']
        for col in required_cols:
            if col in df.columns:
                missing_count = df[col].isna().sum()
                missing_pct = (missing_count / len(df)) * 100
                print(f"{col}: {missing_count} missing values ({missing_pct:.1f}%)")
                
                if missing_pct > 5:
                    print(f"WARNING: High missing value percentage for {col}")
            else:
                print(f"ERROR: Required column {col} not found")
        
        # Statistical summary
        print(f"\nDataset Summary:")
        print(f"Total observations: {len(df)}")
        print(f"Date range: {df['EOM_Dt'].min()} to {df['EOM_Dt'].max()}")
        print(f"Fed Funds range: {df['FEDL01'].min():.2f}% to {df['FEDL01'].max():.2f}%")
        print(f"MMDA rate range: {df['ILMDHYLD'].min():.2f}% to {df['ILMDHYLD'].max():.2f}%")
    
    # =============================================================================
    # DYNAMIC BETA FUNCTIONS
    # =============================================================================
    
    def logistic_beta(self, r, k, m, beta_min, beta_max):
        """
        Standard logistic beta function with bounds enforcement
        
        Parameters:
        -----------
        r : array-like
            Interest rate values
        k : float
            Steepness parameter
        m : float  
            Inflection point
        beta_min, beta_max : float
            Beta bounds
            
        Returns:
        --------
        array-like
            Dynamic beta values
        """
        try:
            exp_term = np.exp(-k * (r - m))
            beta = beta_min + (beta_max - beta_min) / (1 + exp_term)
            return np.clip(beta, beta_min, beta_max)
        except:
            return np.full_like(r, (beta_min + beta_max) / 2)
    
    def volatility_adjusted_beta(self, r, k, m, beta_min, beta_max, vol_ratio, lambda_param):
        """
        Volatility-adjusted logistic beta with stability enhancements
        
        Parameters:
        -----------
        r : array-like
            Interest rate values
        k, m : float
            Logistic parameters  
        beta_min, beta_max : float
            Beta bounds
        vol_ratio : array-like
            Volatility ratio (current/long-run)
        lambda_param : float
            Volatility dampening parameter
            
        Returns:
        --------
        array-like
            Volatility-adjusted dynamic beta values
        """
        try:
            base_beta = self.logistic_beta(r, k, m, beta_min, beta_max)
            vol_adjustment = np.clip(1 - lambda_param * vol_ratio, 0.5, 1.5)
            adjusted_beta = base_beta * vol_adjustment
            return np.clip(adjusted_beta, beta_min * 0.8, beta_max * 1.2)
        except:
            return self.logistic_beta(r, k, m, beta_min, beta_max)
    
    def asymmetric_volatility_beta(self, r, k, m, beta_min, beta_max, vol_ratio, 
                                    lambda_up, lambda_down, rate_change):
        """
        Asymmetric volatility-adjusted beta: different dampening for rising vs falling rates.
        
        This tests whether volatility affects pass-through differently when rates
        are rising (banks may delay increases) vs falling (banks may delay decreases).
        
        Parameters:
        -----------
        r : array-like
            Interest rate values
        k, m : float
            Logistic parameters  
        beta_min, beta_max : float
            Beta bounds
        vol_ratio : array-like
            Volatility ratio (current/long-run)
        lambda_up : float
            Volatility dampening when rates are rising
        lambda_down : float
            Volatility dampening when rates are falling
        rate_change : array-like
            Change in Fed Funds rate (positive = rising, negative = falling)
            
        Returns:
        --------
        array-like
            Asymmetric volatility-adjusted beta values
        """
        try:
            base_beta = self.logistic_beta(r, k, m, beta_min, beta_max)
            
            # Select lambda based on rate direction
            # rate_change > 0 means rates rising, use lambda_up
            # rate_change <= 0 means rates falling/flat, use lambda_down
            lambda_effective = np.where(rate_change > 0, lambda_up, lambda_down)
            
            vol_adjustment = np.clip(1 - lambda_effective * vol_ratio, 0.5, 1.5)
            adjusted_beta = base_beta * vol_adjustment
            return np.clip(adjusted_beta, beta_min * 0.8, beta_max * 1.2)
        except:
            return self.logistic_beta(r, k, m, beta_min, beta_max)
    
    @staticmethod
    def ar_smoothed_beta(beta_unconstrained, max_delta_k, beta_init=None):
        """
        Apply autoregressive smoothing constraint to a beta time series.
        
        Enforces |β̃_t - β̃_{t-1}| ≤ k by clamping changes that exceed the
        tolerance while allowing the smoothed beta to gradually converge to
        the unconstrained equilibrium value.
        
        Parameters:
        -----------
        beta_unconstrained : array-like, shape (T,)
            The "target" beta series from the S-curve + volatility model.
        max_delta_k : float
            Maximum allowable absolute change in beta per period.
            Typical values: 0.01-0.05 (i.e., 1-5 pp per month).
            Set to np.inf to disable smoothing (recover original model).
        beta_init : float, optional
            Starting value. If None, uses first value of beta_unconstrained.
        
        Returns:
        --------
        ndarray, shape (T,)
            Smoothed beta series satisfying the AR constraint.
            
        Notes:
        ------
        The smoothing operation: β̃_t = β̃_{t-1} + clip(β_t^* - β̃_{t-1}, -k, +k)
        
        Properties:
        - β̃_t converges to β_t^* when β_t^* is stable (monotone tracking)
        - Maximum speed of convergence is k per period
        - When k = ∞, β̃_t = β_t^* (no smoothing, original model)
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
    
    def crisis_threshold_beta(self, r, k, m, beta_min, beta_max, vol_ratio, 
                               lambda_normal, lambda_crisis, vol_threshold):
        """
        Crisis threshold volatility beta: different dampening above/below volatility threshold.
        
        This tests whether there's a "regime switch" in how volatility affects
        pass-through when volatility exceeds a crisis threshold (e.g., 90th percentile).
        
        Parameters:
        -----------
        r : array-like
            Interest rate values
        k, m : float
            Logistic parameters  
        beta_min, beta_max : float
            Beta bounds
        vol_ratio : array-like
            Volatility ratio (current/long-run)
        lambda_normal : float
            Volatility dampening in normal periods
        lambda_crisis : float
            Volatility dampening in crisis periods (high volatility)
        vol_threshold : float
            Threshold vol_ratio above which crisis regime applies
            
        Returns:
        --------
        array-like
            Crisis-adjusted beta values
        """
        try:
            base_beta = self.logistic_beta(r, k, m, beta_min, beta_max)
            
            # Select lambda based on volatility level
            # High vol (crisis): may have different (possibly stronger) dampening
            lambda_effective = np.where(vol_ratio > vol_threshold, lambda_crisis, lambda_normal)
            
            vol_adjustment = np.clip(1 - lambda_effective * vol_ratio, 0.5, 1.5)
            adjusted_beta = base_beta * vol_adjustment
            return np.clip(adjusted_beta, beta_min * 0.8, beta_max * 1.2)
        except:
            return self.logistic_beta(r, k, m, beta_min, beta_max)
    
    def quadratic_beta(self, r, a, b, c, beta_min, beta_max):
        """
        Quadratic beta function with strict bounds enforcement
        
        Parameters:
        -----------
        r : array-like
            Interest rate values
        a, b, c : float
            Quadratic coefficients
        beta_min, beta_max : float
            Beta bounds
            
        Returns:
        --------
        array-like
            Bounded quadratic beta values
        """
        try:
            beta = a + b * r + c * r**2
            return np.clip(beta, beta_min, beta_max)
        except:
            return np.full_like(r, (beta_min + beta_max) / 2)
    
    # =============================================================================
    # MODEL ESTIMATION FUNCTIONS
    # =============================================================================
    
    def estimate_enhanced_logistic(self, Y, FEDL01, FHLK3MSPRD, term_spread):
        """
        Enhanced logistic model estimation with robust error handling
        
        Returns:
        --------
        scipy.optimize.OptimizeResult
            Optimization results
        """
        
        def negative_log_likelihood(params):
            try:
                alpha, k, m, beta_min, beta_max, gamma_fhlb, gamma_term = params
                
                beta = self.logistic_beta(FEDL01, k, m, beta_min, beta_max)
                pred = alpha + beta * FEDL01 + gamma_fhlb * FHLK3MSPRD + gamma_term * term_spread
                
                residuals = Y - pred
                sigma_squared = np.var(residuals)
                
                if sigma_squared <= 0 or not np.isfinite(sigma_squared):
                    return 1e10
                    
                n = len(residuals)
                nll = 0.5 * n * np.log(2 * np.pi * sigma_squared) + np.sum(residuals**2) / (2 * sigma_squared)
                
                return nll if np.isfinite(nll) else 1e10
                
            except Exception as e:
                return 1e10
        
        # Parameter bounds: [alpha, k, m, beta_min, beta_max, gamma_fhlb, gamma_term]
        bounds = [(-1, 1), (0.01, 5), (0.5, 5), (0.25, 0.40), (0.50, 0.70), (-2, 2), (-1, 1)]
        initial_params = [0.2, 0.5, 3.0, 0.35, 0.60, 0.3, -0.1]
        
        result = minimize(
            negative_log_likelihood, 
            initial_params, 
            bounds=bounds, 
            method=self.config['optimization_method'],
            options={'maxiter': self.config['max_iterations'], 'ftol': self.config['tolerance']}
        )
        
        return result
    
    def estimate_volatility_adjusted(self, Y, FEDL01, FHLK3MSPRD, term_spread, vol_ratio):
        """
        Volatility-adjusted model estimation with comprehensive error handling
        
        Returns:
        --------
        scipy.optimize.OptimizeResult
            Optimization results
        """
        
        def negative_log_likelihood(params):
            try:
                alpha, k, m, beta_min, beta_max, gamma_fhlb, gamma_term, lambda_param = params
                
                beta = self.volatility_adjusted_beta(
                    FEDL01, k, m, beta_min, beta_max, vol_ratio, lambda_param
                )
                pred = alpha + beta * FEDL01 + gamma_fhlb * FHLK3MSPRD + gamma_term * term_spread
                
                residuals = Y - pred
                sigma_squared = np.var(residuals)
                
                if sigma_squared <= 0 or not np.isfinite(sigma_squared):
                    return 1e10
                    
                n = len(residuals)
                nll = 0.5 * n * np.log(2 * np.pi * sigma_squared) + np.sum(residuals**2) / (2 * sigma_squared)
                
                return nll if np.isfinite(nll) else 1e10
                
            except Exception as e:
                return 1e10
        
        # Parameter bounds: [alpha, k, m, beta_min, beta_max, gamma_fhlb, gamma_term, lambda]
        bounds = [(-1, 1), (0.01, 5), (0.5, 5), (0.25, 0.40), (0.50, 0.70), (-2, 2), (-1, 1), (0, 1)]
        initial_params = [0.2, 0.5, 3.0, 0.35, 0.60, 0.3, -0.1, 0.3]
        
        result = minimize(
            negative_log_likelihood,
            initial_params,
            bounds=bounds,
            method=self.config['optimization_method'],
            options={'maxiter': self.config['max_iterations'], 'ftol': self.config['tolerance']}
        )
        
        return result
    
    def estimate_quadratic(self, Y, FEDL01, FHLK3MSPRD, term_spread):
        """
        Quadratic model estimation for benchmarking
        
        Returns:
        --------
        scipy.optimize.OptimizeResult
            Optimization results
        """
        
        def negative_log_likelihood(params):
            try:
                alpha, a, b, c, beta_min, beta_max, gamma_fhlb, gamma_term = params
                
                beta = self.quadratic_beta(FEDL01, a, b, c, beta_min, beta_max)
                pred = alpha + beta * FEDL01 + gamma_fhlb * FHLK3MSPRD + gamma_term * term_spread
                
                residuals = Y - pred
                sigma_squared = np.var(residuals)
                
                if sigma_squared <= 0 or not np.isfinite(sigma_squared):
                    return 1e10
                    
                n = len(residuals)
                nll = 0.5 * n * np.log(2 * np.pi * sigma_squared) + np.sum(residuals**2) / (2 * sigma_squared)
                
                return nll if np.isfinite(nll) else 1e10
                
            except Exception as e:
                return 1e10
        
        # Parameter bounds: [alpha, a, b, c, beta_min, beta_max, gamma_fhlb, gamma_term]
        bounds = [(-1, 1), (0, 1), (-1, 1), (-0.1, 0.1), (0.25, 0.40), (0.50, 0.70), (-2, 2), (-1, 1)]
        initial_params = [0.2, 0.5, 0.1, 0.01, 0.35, 0.60, 0.3, -0.1]
        
        result = minimize(
            negative_log_likelihood,
            initial_params,
            bounds=bounds,
            method=self.config['optimization_method'],
            options={'maxiter': self.config['max_iterations'], 'ftol': self.config['tolerance']}
        )
        
        return result
    
    def estimate_asymmetric_volatility_revised(self, Y, FEDL01, FHLK3MSPRD, vol_ratio, rate_change):
        """
        RECOMMENDED MODEL: Asymmetric volatility model WITHOUT term spread.
        
        This is the production model specification with 8 parameters. The term spread
        variable was removed due to spurious correlation arising from simultaneous Fed
        policy effects during the 2022-2023 hiking cycle (curve inverted while rates rose).
        
        Parameters:
        -----------
        Y : array-like
            MMDA deposit rate (dependent variable)
        FEDL01 : array-like
            Fed funds effective rate
        FHLK3MSPRD : array-like
            FHLB advance - SOFR spread (bank funding cost)
        vol_ratio : array-like
            Rolling volatility ratio (current / long-term average)
        rate_change : array-like
            Fed funds rate change (for asymmetry detection)
        
        Returns:
        --------
        scipy.optimize.OptimizeResult
            Optimization results with 8 parameters:
            [alpha, k, m, beta_min, beta_max, gamma_fhlb, lambda_up, lambda_down]
        """
        
        def negative_log_likelihood(params):
            try:
                alpha, k, m, beta_min, beta_max, gamma_fhlb, lambda_up, lambda_down = params
                
                beta = self.asymmetric_volatility_beta(
                    FEDL01, k, m, beta_min, beta_max, vol_ratio, 
                    lambda_up, lambda_down, rate_change
                )
                pred = alpha + beta * FEDL01 + gamma_fhlb * FHLK3MSPRD
                
                residuals = Y - pred
                sigma_squared = np.var(residuals)
                
                if sigma_squared <= 0 or not np.isfinite(sigma_squared):
                    return 1e10
                    
                n = len(residuals)
                nll = 0.5 * n * np.log(2 * np.pi * sigma_squared) + np.sum(residuals**2) / (2 * sigma_squared)
                
                return nll if np.isfinite(nll) else 1e10
                
            except Exception as e:
                return 1e10
        
        # Parameter bounds: [alpha, k, m, beta_min, beta_max, gamma_fhlb, lambda_up, lambda_down]
        # 8 parameters - term spread removed
        bounds = [(-1, 1), (0.01, 5), (0.5, 5), (0.30, 0.50), (0.55, 0.80), (-3, 3), (0, 1), (0, 1)]
        initial_params = [0.0, 0.5, 2.5, 0.40, 0.70, 1.0, 0.25, 0.20]
        
        result = minimize(
            negative_log_likelihood,
            initial_params,
            bounds=bounds,
            method=self.config['optimization_method'],
            options={'maxiter': self.config['max_iterations'], 'ftol': self.config['tolerance']}
        )
        
        return result
    
    def estimate_asymmetric_volatility(self, Y, FEDL01, FHLK3MSPRD, term_spread, vol_ratio, rate_change):
        """
        LEGACY: Asymmetric volatility model with term spread (9 parameters).
        
        NOTE: This specification includes term spread which was found to have spurious
        correlation with MMDA rates. Use estimate_asymmetric_volatility_revised() instead.
        
        Returns:
        --------
        scipy.optimize.OptimizeResult
            Optimization results
        """
        
        def negative_log_likelihood(params):
            try:
                alpha, k, m, beta_min, beta_max, gamma_fhlb, gamma_term, lambda_up, lambda_down = params
                
                beta = self.asymmetric_volatility_beta(
                    FEDL01, k, m, beta_min, beta_max, vol_ratio, 
                    lambda_up, lambda_down, rate_change
                )
                pred = alpha + beta * FEDL01 + gamma_fhlb * FHLK3MSPRD + gamma_term * term_spread
                
                residuals = Y - pred
                sigma_squared = np.var(residuals)
                
                if sigma_squared <= 0 or not np.isfinite(sigma_squared):
                    return 1e10
                    
                n = len(residuals)
                nll = 0.5 * n * np.log(2 * np.pi * sigma_squared) + np.sum(residuals**2) / (2 * sigma_squared)
                
                return nll if np.isfinite(nll) else 1e10
                
            except Exception as e:
                return 1e10
        
        # Parameter bounds: [alpha, k, m, beta_min, beta_max, gamma_fhlb, gamma_term, lambda_up, lambda_down]
        # Allow lambda to differ for rising vs falling rates
        bounds = [(-1, 1), (0.01, 5), (0.5, 5), (0.25, 0.40), (0.50, 0.70), (-2, 2), (-1, 1), (0, 1), (0, 1)]
        initial_params = [0.2, 0.5, 3.0, 0.35, 0.60, 0.3, -0.1, 0.3, 0.3]
        
        result = minimize(
            negative_log_likelihood,
            initial_params,
            bounds=bounds,
            method=self.config['optimization_method'],
            options={'maxiter': self.config['max_iterations'], 'ftol': self.config['tolerance']}
        )
        
        return result
    
    def estimate_crisis_threshold(self, Y, FEDL01, FHLK3MSPRD, term_spread, vol_ratio, vol_threshold=None):
        """
        Crisis threshold model: different lambda above/below volatility threshold.
        
        Tests Critique #2: Is there a regime switch at high volatility?
        
        Parameters:
        -----------
        vol_threshold : float, optional
            Volatility ratio threshold. If None, uses 90th percentile.
        
        Returns:
        --------
        scipy.optimize.OptimizeResult
            Optimization results
        """
        
        # Default to 90th percentile of vol_ratio as crisis threshold
        if vol_threshold is None:
            vol_threshold = np.percentile(vol_ratio, 90)
        
        def negative_log_likelihood(params):
            try:
                alpha, k, m, beta_min, beta_max, gamma_fhlb, gamma_term, lambda_normal, lambda_crisis = params
                
                beta = self.crisis_threshold_beta(
                    FEDL01, k, m, beta_min, beta_max, vol_ratio, 
                    lambda_normal, lambda_crisis, vol_threshold
                )
                pred = alpha + beta * FEDL01 + gamma_fhlb * FHLK3MSPRD + gamma_term * term_spread
                
                residuals = Y - pred
                sigma_squared = np.var(residuals)
                
                if sigma_squared <= 0 or not np.isfinite(sigma_squared):
                    return 1e10
                    
                n = len(residuals)
                nll = 0.5 * n * np.log(2 * np.pi * sigma_squared) + np.sum(residuals**2) / (2 * sigma_squared)
                
                return nll if np.isfinite(nll) else 1e10
                
            except Exception as e:
                return 1e10
        
        # Parameter bounds: [alpha, k, m, beta_min, beta_max, gamma_fhlb, gamma_term, lambda_normal, lambda_crisis]
        bounds = [(-1, 1), (0.01, 5), (0.5, 5), (0.25, 0.40), (0.50, 0.70), (-2, 2), (-1, 1), (0, 1), (0, 1)]
        initial_params = [0.2, 0.5, 3.0, 0.35, 0.60, 0.3, -0.1, 0.2, 0.4]
        
        result = minimize(
            negative_log_likelihood,
            initial_params,
            bounds=bounds,
            method=self.config['optimization_method'],
            options={'maxiter': self.config['max_iterations'], 'ftol': self.config['tolerance']}
        )
        
        # Store threshold for later reference
        result.vol_threshold = vol_threshold
        
        return result
    
    # =============================================================================
    # MODEL EVALUATION AND DIAGNOSTICS  
    # =============================================================================
    
    def calculate_model_metrics(self, actual, predicted, n_params):
        """
        Calculate comprehensive model performance metrics
        
        Parameters:
        -----------
        actual : array-like
            Actual values
        predicted : array-like  
            Predicted values
        n_params : int
            Number of model parameters
            
        Returns:
        --------
        dict
            Dictionary of performance metrics
        """
        
        n = len(actual)
        ss_res = np.sum((actual - predicted)**2)
        ss_tot = np.sum((actual - np.mean(actual))**2)
        
        # Basic fit statistics
        r_squared = 1 - (ss_res / ss_tot)
        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - n_params - 1)
        rmse = np.sqrt(ss_res / n)
        mae = np.mean(np.abs(actual - predicted))
        
        # Information criteria  
        mse = ss_res / n
        aic = n * np.log(mse) + 2 * n_params
        bic = n * np.log(mse) + np.log(n) * n_params
        
        return {
            'r_squared': r_squared,
            'adj_r_squared': adj_r_squared,
            'rmse': rmse,
            'mae': mae,
            'aic': aic,
            'bic': bic,
            'n_obs': n,
            'params': n_params
        }
    
    def diagnostic_tests(self, residuals):
        """
        Comprehensive residual diagnostic testing
        
        Parameters:
        -----------
        residuals : array-like
            Model residuals
            
        Returns:
        --------
        dict
            Dictionary of diagnostic test results
        """
        
        try:
            # Jarque-Bera normality test
            jb_result = jarque_bera(residuals)
            jb_pvalue = jb_result[1] if hasattr(jb_result, '__len__') else jb_result.pvalue
            
            # Create design matrix for regression-based tests
            n = len(residuals)
            X = add_constant(np.arange(n))
            
            # Breusch-Godfrey test for serial correlation
            try:
                ols_model = OLS(residuals, X).fit()
                bg_result = acorr_breusch_godfrey(ols_model, nlags=1)
                bg_pvalue = bg_result[1] if hasattr(bg_result, '__len__') else bg_result.pvalue
            except:
                bg_pvalue = np.nan
            
            # White's test for heteroscedasticity
            try:
                white_result = het_white(residuals, X)
                white_pvalue = white_result[1] if hasattr(white_result, '__len__') else white_result.pvalue
            except:
                white_pvalue = np.nan
                
            return {
                'jarque_bera_pvalue': jb_pvalue,
                'breusch_godfrey_pvalue': bg_pvalue,
                'white_pvalue': white_pvalue
            }
            
        except Exception as e:
            print(f"Error in diagnostic tests: {e}")
            return {
                'jarque_bera_pvalue': np.nan,
                'breusch_godfrey_pvalue': np.nan,
                'white_pvalue': np.nan
            }
    
    def stationarity_tests(self, series, name="Series"):
        """
        Comprehensive stationarity testing using ADF and KPSS tests
        
        Parameters:
        -----------
        series : pandas.Series or array-like
            Time series to test
        name : str
            Name of the series for reporting
            
        Returns:
        --------
        dict
            Dictionary of stationarity test results
        """
        
        try:
            # Augmented Dickey-Fuller test
            adf_stat, adf_pvalue = adfuller(series.dropna())[:2]
            
            # KPSS test
            try:
                kpss_stat, kpss_pvalue = kpss(series.dropna(), regression='c')[:2]
            except:
                kpss_stat, kpss_pvalue = np.nan, np.nan
            
            return {
                f'{name}_adf_pvalue': adf_pvalue,
                f'{name}_kpss_pvalue': kpss_pvalue,
                f'{name}_stationary_adf': adf_pvalue < 0.05,
                f'{name}_stationary_kpss': kpss_pvalue > 0.05
            }
            
        except Exception as e:
            print(f"Error in stationarity tests for {name}: {e}")
            return {
                f'{name}_adf_pvalue': np.nan,
                f'{name}_kpss_pvalue': np.nan,
                f'{name}_stationary_adf': False,
                f'{name}_stationary_kpss': False
            }
    
    def comprehensive_stationarity_analysis(self, series_dict):
        """
        Comprehensive stationarity analysis for multiple series including first differences.
        Tests both levels and first differences to determine integration order.
        
        Parameters:
        -----------
        series_dict : dict
            Dictionary mapping series names to their data arrays
            
        Returns:
        --------
        dict
            Comprehensive stationarity results including integration order determination
        """
        
        results = {
            'levels': {},
            'first_differences': {},
            'integration_order': {},
            'summary': []
        }
        
        for name, series in series_dict.items():
            series_clean = pd.Series(series).dropna()
            
            # Test levels
            try:
                adf_level = adfuller(series_clean, autolag='AIC')
                adf_level_stat, adf_level_pvalue = adf_level[0], adf_level[1]
                adf_level_lags = adf_level[2]
                adf_level_crit = adf_level[4]
            except Exception as e:
                print(f"ADF test error for {name} (levels): {e}")
                adf_level_stat, adf_level_pvalue, adf_level_lags = np.nan, np.nan, np.nan
                adf_level_crit = {}
            
            try:
                kpss_level = kpss(series_clean, regression='c', nlags='auto')
                kpss_level_stat, kpss_level_pvalue = kpss_level[0], kpss_level[1]
            except Exception as e:
                print(f"KPSS test error for {name} (levels): {e}")
                kpss_level_stat, kpss_level_pvalue = np.nan, np.nan
            
            results['levels'][name] = {
                'adf_statistic': adf_level_stat,
                'adf_pvalue': adf_level_pvalue,
                'adf_lags': adf_level_lags,
                'adf_critical_values': adf_level_crit,
                'kpss_statistic': kpss_level_stat,
                'kpss_pvalue': kpss_level_pvalue,
                'is_stationary_adf': adf_level_pvalue < 0.05 if not np.isnan(adf_level_pvalue) else False,
                'is_stationary_kpss': kpss_level_pvalue > 0.05 if not np.isnan(kpss_level_pvalue) else False
            }
            
            # Test first differences
            diff_series = series_clean.diff().dropna()
            
            try:
                adf_diff = adfuller(diff_series, autolag='AIC')
                adf_diff_stat, adf_diff_pvalue = adf_diff[0], adf_diff[1]
            except Exception as e:
                print(f"ADF test error for {name} (differences): {e}")
                adf_diff_stat, adf_diff_pvalue = np.nan, np.nan
            
            try:
                kpss_diff = kpss(diff_series, regression='c', nlags='auto')
                kpss_diff_stat, kpss_diff_pvalue = kpss_diff[0], kpss_diff[1]
            except Exception as e:
                print(f"KPSS test error for {name} (differences): {e}")
                kpss_diff_stat, kpss_diff_pvalue = np.nan, np.nan
            
            results['first_differences'][name] = {
                'adf_statistic': adf_diff_stat,
                'adf_pvalue': adf_diff_pvalue,
                'kpss_statistic': kpss_diff_stat,
                'kpss_pvalue': kpss_diff_pvalue,
                'is_stationary_adf': adf_diff_pvalue < 0.05 if not np.isnan(adf_diff_pvalue) else False,
                'is_stationary_kpss': kpss_diff_pvalue > 0.05 if not np.isnan(kpss_diff_pvalue) else False
            }
            
            # Determine integration order
            level_stationary = results['levels'][name]['is_stationary_adf']
            diff_stationary = results['first_differences'][name]['is_stationary_adf']
            
            if level_stationary:
                order = 0  # I(0) - stationary in levels
            elif diff_stationary:
                order = 1  # I(1) - stationary in first differences
            else:
                order = 2  # I(2) or higher - may need further differencing
            
            results['integration_order'][name] = order
            
            results['summary'].append({
                'Variable': name,
                'Level_ADF_pvalue': adf_level_pvalue,
                'Level_Stationary': 'Yes' if level_stationary else 'No',
                'Diff_ADF_pvalue': adf_diff_pvalue,
                'Diff_Stationary': 'Yes' if diff_stationary else 'No',
                'Integration_Order': f'I({order})',
                'Conclusion': 'Stationary' if order == 0 else f'Non-stationary (I({order}))'
            })
        
        return results
    
    def cointegration_test(self, y, x, method='engle-granger'):
        """
        Test for cointegration between two I(1) series using Engle-Granger methodology.
        
        Parameters:
        -----------
        y : array-like
            Dependent variable (e.g., MMDA rate)
        x : array-like
            Independent variable (e.g., Fed Funds rate)
        method : str
            Cointegration test method ('engle-granger')
            
        Returns:
        --------
        dict
            Cointegration test results
        """
        
        results = {}
        
        try:
            # Engle-Granger cointegration test
            # Step 1: Run cointegrating regression y = alpha + beta*x + e
            X_const = add_constant(x)
            coint_reg = OLS(y, X_const).fit()
            
            # Step 2: Test residuals for stationarity
            residuals = coint_reg.resid
            
            # ADF test on residuals (using MacKinnon critical values for cointegration)
            adf_result = adfuller(residuals, autolag='AIC')
            
            # Use statsmodels coint function for proper critical values
            coint_result = coint(y, x, trend='c', autolag='AIC')
            
            results = {
                'method': 'Engle-Granger',
                'coint_statistic': coint_result[0],
                'coint_pvalue': coint_result[1],
                'critical_values': {
                    '1%': coint_result[2][0],
                    '5%': coint_result[2][1],
                    '10%': coint_result[2][2]
                },
                'is_cointegrated': coint_result[1] < 0.05,
                'residual_adf_statistic': adf_result[0],
                'residual_adf_pvalue': adf_result[1],
                'cointegrating_coefficient': coint_reg.params[1],
                'cointegrating_intercept': coint_reg.params[0],
                'interpretation': self._interpret_cointegration(coint_result[1])
            }
            
        except Exception as e:
            print(f"Error in cointegration test: {e}")
            results = {
                'method': 'Engle-Granger',
                'coint_pvalue': np.nan,
                'is_cointegrated': False,
                'error': str(e)
            }
        
        return results
    
    def _interpret_cointegration(self, pvalue):
        """Generate interpretation of cointegration test results"""
        
        if pvalue < 0.01:
            return "Strong evidence of cointegration (p < 0.01). The long-run equilibrium relationship is valid."
        elif pvalue < 0.05:
            return "Evidence of cointegration (p < 0.05). The series share a long-run equilibrium."
        elif pvalue < 0.10:
            return "Weak evidence of cointegration (p < 0.10). Borderline results."
        else:
            return "No evidence of cointegration (p >= 0.10). Series may be spuriously related."
    
    def test_model_residual_stationarity(self, models_dict):
        """
        Test stationarity of model residuals using ADF and KPSS tests.
        
        This is the correct test for model validity: if residuals from our 
        volatility-adjusted model are stationary (I(0)), this indicates the model 
        captures a valid equilibrium relationship, even if the raw series are not 
        cointegrated in the simple bivariate sense.
        
        Parameters:
        -----------
        models_dict : dict
            Dictionary of model results containing predictions
            
        Returns:
        --------
        dict
            Stationarity test results for each model's residuals
        """
        
        import warnings
        warnings.filterwarnings('ignore')
        
        results = {}
        Y = self.results['variables']['Y']
        
        for model_name, model_data in models_dict.items():
            predictions = model_data['predictions']
            residuals = Y - predictions
            n = len(residuals)
            
            try:
                # ADF test (null: unit root exists, i.e., non-stationary)
                adf_result = adfuller(residuals, autolag='AIC')
                
                # KPSS test (null: series is stationary)
                kpss_result = kpss(residuals, regression='c', nlags='auto')
                
                # Determine conclusions
                adf_stationary = adf_result[1] < 0.05  # Reject unit root
                kpss_stationary = kpss_result[0] < kpss_result[3]['5%']  # Fail to reject stationarity
                
                # Unified conclusion
                if adf_stationary and kpss_stationary:
                    conclusion = "STATIONARY (both tests agree)"
                    valid_model = True
                elif not adf_stationary and not kpss_stationary:
                    conclusion = "NON-STATIONARY (both tests agree)"
                    valid_model = False
                else:
                    conclusion = "MIXED (tests disagree, leaning stationary)" if kpss_stationary else "MIXED (tests disagree)"
                    valid_model = kpss_stationary  # KPSS is often more reliable for short samples
                
                results[model_name] = {
                    'n_observations': n,
                    'adf_statistic': adf_result[0],
                    'adf_pvalue': adf_result[1],
                    'adf_critical_1pct': adf_result[4]['1%'],
                    'adf_critical_5pct': adf_result[4]['5%'],
                    'adf_stationary': adf_stationary,
                    'kpss_statistic': kpss_result[0],
                    'kpss_critical_5pct': kpss_result[3]['5%'],
                    'kpss_stationary': kpss_stationary,
                    'conclusion': conclusion,
                    'valid_equilibrium': valid_model,
                    'interpretation': self._interpret_residual_stationarity(adf_stationary, kpss_stationary, model_name)
                }
                
            except Exception as e:
                results[model_name] = {
                    'error': str(e),
                    'conclusion': 'ERROR',
                    'valid_equilibrium': False
                }
        
        return results
    
    def _interpret_residual_stationarity(self, adf_stationary, kpss_stationary, model_name):
        """Generate interpretation of residual stationarity test results"""
        
        if adf_stationary and kpss_stationary:
            return (f"The {model_name.replace('_', ' ')} model residuals are stationary by both ADF and KPSS tests. "
                    f"This confirms the model captures a valid equilibrium relationship between MMDA rates and Fed Funds, "
                    f"validating the use of standard (Newey-West corrected) inference.")
        elif not adf_stationary and not kpss_stationary:
            return (f"The {model_name.replace('_', ' ')} model residuals are non-stationary by both tests. "
                    f"This raises concerns about spurious regression.")
        elif kpss_stationary:
            return (f"Mixed signals: ADF fails to reject unit root (p > 0.05) but KPSS supports stationarity. "
                    f"Given the economic grounding of the model, the KPSS result is weighted more heavily. "
                    f"The model likely captures a valid but imperfect equilibrium.")
        else:
            return (f"Mixed signals: ADF rejects unit root but KPSS rejects stationarity. "
                    f"Further investigation recommended.")
    
    def test_volatility_specifications(self, Y, FEDL01, FHLK3MSPRD, term_spread, vol_ratio, rate_change):
        """
        Comprehensive test of volatility adjustment specifications.
        
        Tests three alternatives:
        1. Symmetric (baseline): Single lambda for all periods
        2. Asymmetric: Different lambda for rising vs falling rates  
        3. Crisis threshold: Different lambda above/below 90th percentile volatility
        
        Returns nested likelihood ratio tests and model comparison metrics.
        
        Parameters:
        -----------
        Y : array-like
            Dependent variable (MMDA rate)
        FEDL01 : array-like
            Fed Funds rate
        FHLK3MSPRD : array-like
            FHLB spread
        term_spread : array-like
            Term spread (1Y-3M)
        vol_ratio : array-like
            Volatility ratio
        rate_change : array-like
            Change in Fed Funds rate
            
        Returns:
        --------
        dict
            Comprehensive comparison results
        """
        
        results = {
            'models': {},
            'comparisons': {},
            'recommendation': None
        }
        
        n = len(Y)
        vol_90pct = np.percentile(vol_ratio, 90)
        
        print("\n  Testing Volatility Specifications (Critique #2):")
        print("  " + "-" * 50)
        
        # Model 1: Symmetric (baseline) - already estimated, re-estimate for consistent comparison
        print("    1. Symmetric volatility adjustment (baseline)...")
        try:
            result_sym = self.estimate_volatility_adjusted(Y, FEDL01, FHLK3MSPRD, term_spread, vol_ratio)
            params_sym = result_sym.x
            
            beta_sym = self.volatility_adjusted_beta(
                FEDL01, params_sym[1], params_sym[2], params_sym[3], 
                params_sym[4], vol_ratio, params_sym[7]
            )
            pred_sym = params_sym[0] + beta_sym * FEDL01 + params_sym[5] * FHLK3MSPRD + params_sym[6] * term_spread
            
            ll_sym = -result_sym.fun
            rmse_sym = np.sqrt(np.mean((Y - pred_sym)**2))
            n_params_sym = 8
            aic_sym = -2 * ll_sym + 2 * n_params_sym
            bic_sym = -2 * ll_sym + np.log(n) * n_params_sym
            
            results['models']['symmetric'] = {
                'success': result_sym.success,
                'log_likelihood': ll_sym,
                'n_params': n_params_sym,
                'aic': aic_sym,
                'bic': bic_sym,
                'rmse': rmse_sym,
                'lambda': params_sym[7],
                'predictions': pred_sym,
                'params': dict(zip(['alpha', 'k', 'm', 'beta_min', 'beta_max', 'gamma_fhlb', 'gamma_term', 'lambda'], params_sym))
            }
            print(f"       λ = {params_sym[7]:.4f}, RMSE = {rmse_sym:.4f}%, AIC = {aic_sym:.1f}")
            
        except Exception as e:
            print(f"       ERROR: {e}")
            results['models']['symmetric'] = {'success': False, 'error': str(e)}
        
        # Model 2: Asymmetric - different lambda for rising vs falling rates
        print("    2. Asymmetric volatility (rising vs falling rates)...")
        try:
            result_asym = self.estimate_asymmetric_volatility(Y, FEDL01, FHLK3MSPRD, term_spread, vol_ratio, rate_change)
            params_asym = result_asym.x
            
            beta_asym = self.asymmetric_volatility_beta(
                FEDL01, params_asym[1], params_asym[2], params_asym[3], 
                params_asym[4], vol_ratio, params_asym[7], params_asym[8], rate_change
            )
            pred_asym = params_asym[0] + beta_asym * FEDL01 + params_asym[5] * FHLK3MSPRD + params_asym[6] * term_spread
            
            ll_asym = -result_asym.fun
            rmse_asym = np.sqrt(np.mean((Y - pred_asym)**2))
            n_params_asym = 9
            aic_asym = -2 * ll_asym + 2 * n_params_asym
            bic_asym = -2 * ll_asym + np.log(n) * n_params_asym
            
            results['models']['asymmetric'] = {
                'success': result_asym.success,
                'log_likelihood': ll_asym,
                'n_params': n_params_asym,
                'aic': aic_asym,
                'bic': bic_asym,
                'rmse': rmse_asym,
                'lambda_up': params_asym[7],
                'lambda_down': params_asym[8],
                'predictions': pred_asym,
                'params': dict(zip(['alpha', 'k', 'm', 'beta_min', 'beta_max', 'gamma_fhlb', 'gamma_term', 'lambda_up', 'lambda_down'], params_asym))
            }
            print(f"       λ_up = {params_asym[7]:.4f}, λ_down = {params_asym[8]:.4f}, RMSE = {rmse_asym:.4f}%, AIC = {aic_asym:.1f}")
            
        except Exception as e:
            print(f"       ERROR: {e}")
            results['models']['asymmetric'] = {'success': False, 'error': str(e)}
        
        # Model 3: Crisis threshold - different lambda above/below 90th percentile
        print(f"    3. Crisis threshold (vol_ratio > {vol_90pct:.2f} = 90th percentile)...")
        try:
            result_crisis = self.estimate_crisis_threshold(Y, FEDL01, FHLK3MSPRD, term_spread, vol_ratio, vol_90pct)
            params_crisis = result_crisis.x
            
            beta_crisis = self.crisis_threshold_beta(
                FEDL01, params_crisis[1], params_crisis[2], params_crisis[3], 
                params_crisis[4], vol_ratio, params_crisis[7], params_crisis[8], vol_90pct
            )
            pred_crisis = params_crisis[0] + beta_crisis * FEDL01 + params_crisis[5] * FHLK3MSPRD + params_crisis[6] * term_spread
            
            ll_crisis = -result_crisis.fun
            rmse_crisis = np.sqrt(np.mean((Y - pred_crisis)**2))
            n_params_crisis = 9
            aic_crisis = -2 * ll_crisis + 2 * n_params_crisis
            bic_crisis = -2 * ll_crisis + np.log(n) * n_params_crisis
            
            # Count observations in each regime
            n_crisis = np.sum(vol_ratio > vol_90pct)
            n_normal = np.sum(vol_ratio <= vol_90pct)
            
            results['models']['crisis_threshold'] = {
                'success': result_crisis.success,
                'log_likelihood': ll_crisis,
                'n_params': n_params_crisis,
                'aic': aic_crisis,
                'bic': bic_crisis,
                'rmse': rmse_crisis,
                'lambda_normal': params_crisis[7],
                'lambda_crisis': params_crisis[8],
                'vol_threshold': vol_90pct,
                'n_normal': n_normal,
                'n_crisis': n_crisis,
                'predictions': pred_crisis,
                'params': dict(zip(['alpha', 'k', 'm', 'beta_min', 'beta_max', 'gamma_fhlb', 'gamma_term', 'lambda_normal', 'lambda_crisis'], params_crisis))
            }
            print(f"       λ_normal = {params_crisis[7]:.4f}, λ_crisis = {params_crisis[8]:.4f}, RMSE = {rmse_crisis:.4f}%, AIC = {aic_crisis:.1f}")
            print(f"       Regime split: {n_normal} normal, {n_crisis} crisis observations")
            
        except Exception as e:
            print(f"       ERROR: {e}")
            results['models']['crisis_threshold'] = {'success': False, 'error': str(e)}
        
        # Likelihood Ratio Tests (nested models)
        print("\n    Likelihood Ratio Tests:")
        
        # Symmetric vs Asymmetric (testing if lambda_up != lambda_down)
        if results['models'].get('symmetric', {}).get('success') and results['models'].get('asymmetric', {}).get('success'):
            ll_restricted = results['models']['symmetric']['log_likelihood']
            ll_unrestricted = results['models']['asymmetric']['log_likelihood']
            lr_stat = 2 * (ll_unrestricted - ll_restricted)
            df = 1  # One additional parameter
            from scipy.stats import chi2
            p_value = 1 - chi2.cdf(lr_stat, df)
            
            results['comparisons']['symmetric_vs_asymmetric'] = {
                'lr_statistic': lr_stat,
                'df': df,
                'p_value': p_value,
                'reject_null': p_value < 0.05,
                'interpretation': 'Asymmetry IS significant' if p_value < 0.05 else 'Asymmetry NOT significant'
            }
            print(f"       Symmetric vs Asymmetric: LR = {lr_stat:.2f}, df = {df}, p = {p_value:.4f}")
            print(f"       >>> {'REJECT' if p_value < 0.05 else 'FAIL TO REJECT'} symmetry (at 5%)")
        
        # Symmetric vs Crisis (testing if lambda_normal != lambda_crisis)
        if results['models'].get('symmetric', {}).get('success') and results['models'].get('crisis_threshold', {}).get('success'):
            ll_restricted = results['models']['symmetric']['log_likelihood']
            ll_unrestricted = results['models']['crisis_threshold']['log_likelihood']
            lr_stat = 2 * (ll_unrestricted - ll_restricted)
            df = 1
            p_value = 1 - chi2.cdf(lr_stat, df)
            
            results['comparisons']['symmetric_vs_crisis'] = {
                'lr_statistic': lr_stat,
                'df': df,
                'p_value': p_value,
                'reject_null': p_value < 0.05,
                'interpretation': 'Crisis threshold IS significant' if p_value < 0.05 else 'Crisis threshold NOT significant'
            }
            print(f"       Symmetric vs Crisis: LR = {lr_stat:.2f}, df = {df}, p = {p_value:.4f}")
            print(f"       >>> {'REJECT' if p_value < 0.05 else 'FAIL TO REJECT'} single lambda (at 5%)")
        
        # Model selection by AIC/BIC
        print("\n    Model Selection (AIC/BIC):")
        valid_models = {k: v for k, v in results['models'].items() if v.get('success', False)}
        
        if valid_models:
            best_aic = min(valid_models.items(), key=lambda x: x[1]['aic'])
            best_bic = min(valid_models.items(), key=lambda x: x[1]['bic'])
            
            results['best_by_aic'] = best_aic[0]
            results['best_by_bic'] = best_bic[0]
            
            print(f"       Best by AIC: {best_aic[0]} (AIC = {best_aic[1]['aic']:.1f})")
            print(f"       Best by BIC: {best_bic[0]} (BIC = {best_bic[1]['bic']:.1f})")
            
            # Generate recommendation
            asym_test = results['comparisons'].get('symmetric_vs_asymmetric', {})
            crisis_test = results['comparisons'].get('symmetric_vs_crisis', {})
            
            if not asym_test.get('reject_null', False) and not crisis_test.get('reject_null', False):
                results['recommendation'] = 'symmetric'
                results['recommendation_reason'] = ('Neither asymmetry nor crisis threshold significantly improves fit. '
                                                     'The parsimonious symmetric model is preferred.')
            elif asym_test.get('reject_null', False) and not crisis_test.get('reject_null', False):
                results['recommendation'] = 'asymmetric'
                results['recommendation_reason'] = ('Asymmetric model significantly improves fit over symmetric. '
                                                     'Volatility affects pass-through differently in rising vs falling rate environments.')
            elif crisis_test.get('reject_null', False) and not asym_test.get('reject_null', False):
                results['recommendation'] = 'crisis_threshold'
                results['recommendation_reason'] = ('Crisis threshold model significantly improves fit. '
                                                     'There is a regime switch in volatility effects above the 90th percentile.')
            else:
                # Both significant - choose by BIC (penalizes complexity more)
                results['recommendation'] = best_bic[0]
                results['recommendation_reason'] = ('Both asymmetric and crisis models improve fit. '
                                                     f'Selecting {best_bic[0]} based on BIC.')
            
            print(f"\n    RECOMMENDATION: {results['recommendation'].upper()}")
            print(f"       {results['recommendation_reason']}")
        
        return results
    
    def test_sigma_star_sensitivity(self, Y, FEDL01, FHLK3MSPRD, term_spread):
        """
        Test sensitivity of model to σ* (long-run volatility) calculation window.
        
        Tests 3-year (36 months) vs 5-year (60 months) expanding windows for 
        calculating long-run average volatility.
        
        Parameters:
        -----------
        Y : array-like
            MMDA rate
        FEDL01 : array-like
            Fed Funds rate
        FHLK3MSPRD : array-like
            FHLB spread
        term_spread : array-like
            Term spread
            
        Returns:
        --------
        dict
            Sensitivity analysis results
        """
        
        results = {
            'windows': {},
            'comparison': {}
        }
        
        print("\n    Testing σ* window sensitivity:")
        print("    " + "-" * 50)
        
        # Calculate rate changes from original data
        rate_changes = pd.Series(FEDL01).diff().fillna(0).values
        
        # Test different rolling windows for volatility calculation
        windows_to_test = {
            '24m (baseline)': 24,
            '36m (3-year)': 36,
            '48m (4-year)': 48
        }
        
        n = len(FEDL01)
        
        for window_name, window_size in windows_to_test.items():
            print(f"      Testing {window_name} window...")
            
            # Recalculate volatility with different window
            vol_series = pd.Series(rate_changes).rolling(
                window=window_size, 
                min_periods=12
            ).std()
            
            # Fill initial NaNs with first valid value
            first_valid = vol_series.first_valid_index()
            if first_valid is not None:
                vol_series = vol_series.fillna(vol_series.loc[first_valid])
            
            # Calculate vol_star as expanding mean
            vol_star_series = vol_series.expanding().mean()
            vol_ratio_new = (vol_series / vol_star_series).values
            
            # Estimate volatility-adjusted model with new vol_ratio
            try:
                result = self.estimate_volatility_adjusted(Y, FEDL01, FHLK3MSPRD, term_spread, vol_ratio_new)
                params = result.x
                
                # Calculate predictions and metrics
                beta = self.volatility_adjusted_beta(
                    FEDL01, params[1], params[2], params[3], params[4], vol_ratio_new, params[7]
                )
                pred = params[0] + beta * FEDL01 + params[5] * FHLK3MSPRD + params[6] * term_spread
                
                rmse = np.sqrt(np.mean((Y - pred)**2))
                ll = -result.fun
                aic = -2 * ll + 2 * 8  # 8 parameters
                bic = -2 * ll + np.log(n) * 8
                
                results['windows'][window_name] = {
                    'window_size': window_size,
                    'success': result.success,
                    'lambda': params[7],
                    'log_likelihood': ll,
                    'rmse': rmse,
                    'aic': aic,
                    'bic': bic,
                    'vol_ratio_mean': np.mean(vol_ratio_new),
                    'vol_ratio_std': np.std(vol_ratio_new)
                }
                
                print(f"         λ = {params[7]:.4f}, RMSE = {rmse:.4f}%, AIC = {aic:.1f}")
                
            except Exception as e:
                results['windows'][window_name] = {
                    'window_size': window_size,
                    'success': False,
                    'error': str(e)
                }
                print(f"         ERROR: {e}")
        
        # Compare results
        valid_windows = {k: v for k, v in results['windows'].items() if v.get('success', False)}
        
        if len(valid_windows) >= 2:
            # Find best by AIC
            best_aic = min(valid_windows.items(), key=lambda x: x[1]['aic'])
            best_bic = min(valid_windows.items(), key=lambda x: x[1]['bic'])
            
            results['comparison']['best_by_aic'] = best_aic[0]
            results['comparison']['best_by_bic'] = best_bic[0]
            
            # Calculate range of lambda estimates across windows
            lambdas = [v['lambda'] for v in valid_windows.values()]
            results['comparison']['lambda_range'] = {
                'min': min(lambdas),
                'max': max(lambdas),
                'spread': max(lambdas) - min(lambdas)
            }
            
            # Sensitivity assessment
            lambda_spread = max(lambdas) - min(lambdas)
            if lambda_spread < 0.05:
                sensitivity = 'LOW'
                interpretation = 'λ estimates are robust to σ* window choice (spread < 5%)'
            elif lambda_spread < 0.10:
                sensitivity = 'MODERATE'
                interpretation = 'λ estimates show some sensitivity to σ* window (spread 5-10%)'
            else:
                sensitivity = 'HIGH'
                interpretation = 'λ estimates are sensitive to σ* window choice (spread > 10%)'
            
            results['comparison']['sensitivity'] = sensitivity
            results['comparison']['interpretation'] = interpretation
            
            print(f"\n    σ* SENSITIVITY RESULTS:")
            print(f"       Best window by AIC: {best_aic[0]}")
            print(f"       Best window by BIC: {best_bic[0]}")
            print(f"       λ range: {min(lambdas):.4f} to {max(lambdas):.4f} (spread = {lambda_spread:.4f})")
            print(f"       Sensitivity: {sensitivity} - {interpretation}")
        
        return results
    
    def test_regime_specific_rmse(self, models=None):
        """
        Calculate regime-specific RMSE to assess model performance across different
        rate environments (Critique #3).
        
        Tests model performance in:
        1. Low rate regime (Fed Funds < 1%, approximately 2020-2022)
        2. Rate decline period (Fed Funds declining, 2019-2020)
        3. Rate increase period (Fed Funds rising, 2022-2025)
        4. Below inflection point (Fed Funds < ~3%)
        5. Above inflection point (Fed Funds >= ~3%)
        
        Parameters:
        -----------
        models : dict, optional
            Dictionary of model results. If None, uses self.models
            
        Returns:
        --------
        dict
            Regime-specific performance metrics for each model
        """
        
        if models is None:
            models = self.models
            
        if not models:
            raise ValueError("No models available. Run analysis first.")
            
        if self.data is None:
            raise ValueError("Data not loaded.")
        
        results = {
            'regimes': {},
            'model_comparison': {},
            'interpretation': {}
        }
        
        # Extract data
        dates = pd.to_datetime(self.data['EOM_Dt'])
        fed_funds = self.data['FEDL01'].values
        actual = self.data['ILMDHYLD'].values
        
        # Calculate rate changes for regime identification
        ff_change = pd.Series(fed_funds).diff().fillna(0).values
        
        # Get inflection point from vol_adjusted model if available
        inflection_point = 3.0  # default
        if 'vol_adjusted' in models and 'params' in models['vol_adjusted']:
            inflection_point = models['vol_adjusted']['params'].get('m', 3.0)
        
        print(f"\n    Inflection point (m): {inflection_point:.2f}%")
        
        # Define regimes
        regimes = {
            'low_rate': {
                'description': 'Low Rate Environment (Fed Funds < 1%)',
                'mask': fed_funds < 1.0,
                'period': 'Approximately 2020-2022'
            },
            'rate_decline': {
                'description': 'Rate Decline Period (Fed Funds declining)',
                'mask': (dates >= '2019-07-01') & (dates <= '2020-06-30'),
                'period': 'July 2019 - June 2020'
            },
            'rate_increase': {
                'description': 'Rate Increase Period (Rising rates)',
                'mask': (dates >= '2022-03-01') & (dates <= '2025-03-31'),
                'period': 'March 2022 - March 2025'
            },
            'below_inflection': {
                'description': f'Below Inflection Point (Fed Funds < {inflection_point:.1f}%)',
                'mask': fed_funds < inflection_point,
                'period': 'Rate-dependent'
            },
            'above_inflection': {
                'description': f'Above Inflection Point (Fed Funds >= {inflection_point:.1f}%)',
                'mask': fed_funds >= inflection_point,
                'period': 'Rate-dependent'
            },
            'full_sample': {
                'description': 'Full Sample',
                'mask': np.ones(len(fed_funds), dtype=bool),
                'period': '2017-2025'
            }
        }
        
        print("\n    Regime Definitions:")
        print("    " + "-" * 60)
        for regime_name, regime_info in regimes.items():
            n_obs = regime_info['mask'].sum()
            if n_obs > 0:
                ff_range = f"{fed_funds[regime_info['mask']].min():.2f}% - {fed_funds[regime_info['mask']].max():.2f}%"
            else:
                ff_range = "N/A"
            print(f"    {regime_name}: {n_obs} obs, Fed Funds range: {ff_range}")
        
        # Calculate RMSE for each model in each regime
        print("\n    Calculating Regime-Specific RMSE...")
        print("    " + "-" * 60)
        
        for model_name, model in models.items():
            if 'predictions' not in model:
                continue
                
            predictions = model['predictions']
            
            results['regimes'][model_name] = {}
            
            for regime_name, regime_info in regimes.items():
                mask = regime_info['mask']
                n_obs = mask.sum()
                
                if n_obs > 0:
                    regime_actual = actual[mask]
                    regime_pred = predictions[mask]
                    
                    rmse = np.sqrt(np.mean((regime_actual - regime_pred)**2))
                    mae = np.mean(np.abs(regime_actual - regime_pred))
                    mean_error = np.mean(regime_pred - regime_actual)  # bias
                    max_error = np.max(np.abs(regime_actual - regime_pred))
                    
                    results['regimes'][model_name][regime_name] = {
                        'n_obs': n_obs,
                        'rmse': rmse,
                        'mae': mae,
                        'mean_error': mean_error,
                        'max_error': max_error,
                        'fed_funds_mean': fed_funds[mask].mean(),
                        'fed_funds_range': (fed_funds[mask].min(), fed_funds[mask].max())
                    }
        
        # Print results table
        print(f"\n    {'Model':<20} {'Regime':<25} {'N':<5} {'RMSE':>8} {'MAE':>8} {'Bias':>8}")
        print("    " + "-" * 80)
        
        for model_name in results['regimes'].keys():
            for regime_name in results['regimes'][model_name].keys():
                metrics = results['regimes'][model_name][regime_name]
                print(f"    {model_name:<20} {regime_name:<25} {metrics['n_obs']:<5} "
                      f"{metrics['rmse']:>8.4f} {metrics['mae']:>8.4f} {metrics['mean_error']:>+8.4f}")
        
        # Calculate relative performance (vs full sample RMSE)
        print("\n    Relative Performance (RMSE / Full Sample RMSE):")
        print("    " + "-" * 60)
        
        for model_name in results['regimes'].keys():
            full_rmse = results['regimes'][model_name]['full_sample']['rmse']
            print(f"\n    {model_name.upper()}:")
            
            for regime_name, metrics in results['regimes'][model_name].items():
                if regime_name != 'full_sample':
                    rel_perf = metrics['rmse'] / full_rmse
                    status = "✓" if rel_perf <= 1.5 else "⚠" if rel_perf <= 2.0 else "✗"
                    print(f"      {regime_name}: {rel_perf:.2f}x {status}")
                    
                    results['regimes'][model_name][regime_name]['relative_rmse'] = rel_perf
        
        # Model comparison across regimes
        if len(results['regimes']) > 1 and 'vol_adjusted' in results['regimes']:
            print("\n    Model Comparison (Vol-Adjusted vs Others):")
            print("    " + "-" * 60)
            
            vol_results = results['regimes']['vol_adjusted']
            
            for model_name, model_results in results['regimes'].items():
                if model_name != 'vol_adjusted':
                    results['model_comparison'][f'{model_name}_vs_vol_adjusted'] = {}
                    
                    print(f"\n    {model_name.upper()} vs VOL_ADJUSTED:")
                    for regime_name in vol_results.keys():
                        if regime_name in model_results:
                            vol_rmse = vol_results[regime_name]['rmse']
                            other_rmse = model_results[regime_name]['rmse']
                            improvement = (other_rmse - vol_rmse) / other_rmse * 100
                            
                            results['model_comparison'][f'{model_name}_vs_vol_adjusted'][regime_name] = {
                                'vol_adjusted_rmse': vol_rmse,
                                'other_rmse': other_rmse,
                                'improvement_pct': improvement
                            }
                            
                            status = "✓ better" if improvement > 0 else "✗ worse"
                            print(f"      {regime_name}: {improvement:+.1f}% {status}")
        
        # Generate interpretation
        if 'vol_adjusted' in results['regimes']:
            vol_results = results['regimes']['vol_adjusted']
            
            interpretations = []
            
            # Check low rate performance
            if 'low_rate' in vol_results:
                low_rel = vol_results['low_rate'].get('relative_rmse', 0)
                if low_rel > 1.5:
                    interpretations.append(f"Model shows elevated errors in low-rate regime ({low_rel:.2f}x full sample)")
                else:
                    interpretations.append(f"Model performs well in low-rate regime ({low_rel:.2f}x full sample)")
            
            # Check rate increase performance
            if 'rate_increase' in vol_results:
                inc_rel = vol_results['rate_increase'].get('relative_rmse', 0)
                if inc_rel > 1.5:
                    interpretations.append(f"Model shows elevated errors during rate increases ({inc_rel:.2f}x full sample)")
                else:
                    interpretations.append(f"Model captures rate increase dynamics well ({inc_rel:.2f}x full sample)")
            
            # Check inflection point asymmetry
            if 'below_inflection' in vol_results and 'above_inflection' in vol_results:
                below_rmse = vol_results['below_inflection']['rmse']
                above_rmse = vol_results['above_inflection']['rmse']
                
                if below_rmse > above_rmse * 1.3:
                    interpretations.append(f"Higher errors below inflection ({below_rmse:.4f}% vs {above_rmse:.4f}% above) - "
                                         "consistent with downward stickiness")
                elif above_rmse > below_rmse * 1.3:
                    interpretations.append(f"Higher errors above inflection ({above_rmse:.4f}% vs {below_rmse:.4f}% below)")
                else:
                    interpretations.append("Balanced performance above/below inflection point")
            
            # Check for bias
            if 'rate_increase' in vol_results:
                bias = vol_results['rate_increase']['mean_error']
                if abs(bias) > 0.05:
                    direction = "over-predicting" if bias > 0 else "under-predicting"
                    interpretations.append(f"Systematic {direction} during rate increases (bias = {bias:+.4f}%)")
            
            results['interpretation'] = interpretations
            
            print("\n    INTERPRETATION:")
            print("    " + "-" * 60)
            for interp in interpretations:
                print(f"    • {interp}")
        
        return results

    def calculate_newey_west_se(self, residuals, X, n_lags=None):
        """
        Calculate Newey-West HAC (Heteroscedasticity and Autocorrelation Consistent) 
        standard errors for robust inference.
        
        Parameters:
        -----------
        residuals : array-like
            Model residuals
        X : array-like
            Design matrix (regressors)
        n_lags : int, optional
            Number of lags for HAC. If None, uses Newey-West optimal lag selection.
            
        Returns:
        --------
        dict
            Dictionary containing HAC covariance matrix and robust standard errors
        """
        
        try:
            n = len(residuals)
            k = X.shape[1] if len(X.shape) > 1 else 1
            
            # Newey-West optimal lag selection: floor(4*(n/100)^(2/9))
            if n_lags is None:
                n_lags = int(np.floor(4 * (n / 100) ** (2/9)))
            
            # Convert to numpy arrays
            e = np.array(residuals).flatten()
            X = np.array(X)
            if len(X.shape) == 1:
                X = X.reshape(-1, 1)
            
            # Calculate X'X inverse
            XtX_inv = np.linalg.inv(X.T @ X)
            
            # Initialize S_0 (heteroscedasticity-consistent component)
            # S_0 = (1/n) * sum(e_t^2 * x_t * x_t')
            S = np.zeros((k, k))
            for t in range(n):
                x_t = X[t, :].reshape(-1, 1)
                S += (e[t] ** 2) * (x_t @ x_t.T)
            
            # Add autocorrelation components with Bartlett kernel weights
            for lag in range(1, n_lags + 1):
                weight = 1 - lag / (n_lags + 1)  # Bartlett kernel
                for t in range(lag, n):
                    x_t = X[t, :].reshape(-1, 1)
                    x_t_lag = X[t - lag, :].reshape(-1, 1)
                    # Add both e_t*e_{t-lag}*x_t*x_{t-lag}' and its transpose
                    cross_term = e[t] * e[t - lag] * (x_t @ x_t_lag.T + x_t_lag @ x_t.T)
                    S += weight * cross_term
            
            # HAC covariance matrix: (X'X)^{-1} * S * (X'X)^{-1}
            hac_cov = XtX_inv @ S @ XtX_inv
            
            # Robust standard errors are square roots of diagonal elements
            robust_se = np.sqrt(np.diag(hac_cov))
            
            return {
                'hac_covariance': hac_cov,
                'robust_standard_errors': robust_se,
                'n_lags': n_lags,
                'n_observations': n,
                'n_parameters': k
            }
            
        except Exception as e:
            print(f"Error calculating Newey-West SE: {e}")
            return {
                'hac_covariance': None,
                'robust_standard_errors': None,
                'error': str(e)
            }
    
    def sandwich_standard_errors(self, model_name='asym_vol_revised', n_lags=None, 
                                  eps=1e-5, max_beta_change=None):
        """
        Compute sandwich (Huber-White) standard errors for ALL model parameters.
        
        The sandwich estimator properly accounts for serial dependence in the
        time series, unlike the naive Hessian-based MLE variance which assumes
        independent observations.
        
        V_sandwich = H^{-1} · S · H^{-1}
        
        where:
            H = observed information matrix (Hessian of NLL)
            S = long-run variance of score vectors (Newey-West HAC)
            g_t = per-observation score vector (gradient of individual log-likelihood)
        
        Parameters:
        -----------
        model_name : str
            Which model to compute SEs for. Default: 'asym_vol_revised' (production).
        n_lags : int, optional
            Newey-West truncation lag. If None, uses optimal lag selection.
        eps : float
            Step size for numerical differentiation.
        max_beta_change : float, optional
            If provided, applies AR smoothing constraint to beta before computing.
            
        Returns:
        --------
        dict
            Contains sandwich_se, hessian_se, sandwich_cov, param_table, etc.
        """
        from enhanced_dynamic_beta_model import sandwich_standard_errors as _sandwich_se
        
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Run analysis first.")
        
        model_data = self.models[model_name]
        
        # Extract data arrays
        Y = self.results['variables']['Y']
        FEDL01 = self.results['variables']['FEDL01']
        FHLK3MSPRD = self.results['variables']['FHLK3MSPRD']
        vol_ratio = self.data['vol_ratio'].values
        rate_change = self.data['FEDL01_change'].fillna(0).values
        
        # Determine parameter names and NLL functions based on model type
        if 'revised' in model_name or model_name == 'asym_vol_revised':
            param_names = ['alpha', 'k', 'm', 'beta_min', 'beta_max', 
                          'gamma_fhlb', 'lambda_up', 'lambda_down']
            
            def nll_per_obs(params, Y, FEDL01, FHLK3MSPRD, vol_ratio, rate_change):
                alpha, k, m, beta_min, beta_max, gamma_fhlb, lambda_up, lambda_down = params
                beta = self.asymmetric_volatility_beta(
                    FEDL01, k, m, beta_min, beta_max, vol_ratio,
                    lambda_up, lambda_down, rate_change
                )
                if max_beta_change is not None:
                    beta = self.ar_smoothed_beta(beta, max_beta_change)
                pred = alpha + beta * FEDL01 + gamma_fhlb * FHLK3MSPRD
                resid = Y - pred
                sigma_sq = np.var(resid)
                if sigma_sq <= 0 or not np.isfinite(sigma_sq):
                    return np.full(len(Y), 1e10 / len(Y))
                return 0.5 * np.log(2 * np.pi * sigma_sq) + resid**2 / (2 * sigma_sq)
            
            def nll_total(params, Y, FEDL01, FHLK3MSPRD, vol_ratio, rate_change):
                return np.sum(nll_per_obs(params, Y, FEDL01, FHLK3MSPRD, vol_ratio, rate_change))
            
            data_args = (Y, FEDL01, FHLK3MSPRD, vol_ratio, rate_change)
        else:
            raise ValueError(f"Sandwich SEs not implemented for model type '{model_name}'.")
        
        params_hat = model_data['result'].x
        
        print(f"Computing sandwich standard errors for '{model_name}' model...")
        result = _sandwich_se(
            nll_per_obs_fn=nll_per_obs,
            nll_total_fn=nll_total,
            params_hat=params_hat,
            data_args=data_args,
            n_lags=n_lags,
            eps=eps,
            param_names=param_names
        )
        
        # Store in model data
        model_data['sandwich_se'] = result
        
        return result
    
    def robust_inference(self, model_name='vol_adjusted'):
        """
        Perform robust inference using Newey-West standard errors for a fitted model.
        
        Parameters:
        -----------
        model_name : str
            Name of the model to analyze
            
        Returns:
        --------
        dict
            Robust inference results including t-statistics and p-values
        """
        
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Run full_analysis first.")
        
        model_data = self.models[model_name]
        params = model_data['params']
        residuals = self.results['variables']['Y'] - model_data['predictions']
        
        # Build design matrix based on model type
        Y = self.results['variables']['Y']
        FEDL01 = self.results['variables']['FEDL01']
        FHLK3MSPRD = self.results['variables']['FHLK3MSPRD']
        term_spread = self.results['variables']['term_spread']
        
        # For robust inference, we use a linearized approximation
        # around the estimated parameters
        n = len(Y)
        beta_values = model_data['beta']
        
        # Design matrix: [1, beta*FEDL01, FHLK3MSPRD, term_spread]
        # Note: This is a simplified linearization; full inference would require
        # numerical derivatives of the nonlinear parameters
        X = np.column_stack([
            np.ones(n),
            beta_values * FEDL01,
            FHLK3MSPRD,
            term_spread
        ])
        
        # Calculate Newey-West standard errors
        nw_results = self.calculate_newey_west_se(residuals, X)
        
        if nw_results['robust_standard_errors'] is not None:
            # Extract the linear parameters for which we can compute robust inference
            linear_params = ['alpha', 'gamma_fhlb', 'gamma_term']
            linear_estimates = [params.get('alpha', 0), params.get('gamma_fhlb', 0), params.get('gamma_term', 0)]
            
            # Map robust SEs to parameters (indices 0, 2, 3 in design matrix)
            robust_se = nw_results['robust_standard_errors']
            
            inference_results = {
                'model': model_name,
                'n_lags_used': nw_results['n_lags'],
                'parameters': {}
            }
            
            # Compute t-statistics and p-values for linear parameters
            param_indices = {'alpha': 0, 'gamma_fhlb': 2, 'gamma_term': 3}
            for param, idx in param_indices.items():
                if param in params and idx < len(robust_se):
                    estimate = params[param]
                    se = robust_se[idx]
                    t_stat = estimate / se if se > 0 else np.nan
                    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - len(robust_se)))
                    
                    inference_results['parameters'][param] = {
                        'estimate': estimate,
                        'robust_se': se,
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'significant_5pct': p_value < 0.05 if not np.isnan(p_value) else False
                    }
            
            # Add note about nonlinear parameters
            inference_results['note'] = (
                "Robust standard errors computed for linear parameters using Newey-West HAC. "
                "For FULL robust inference on ALL parameters (including nonlinear k, m, "
                "beta_min, beta_max, lambda), use sandwich_standard_errors() method which "
                "computes the Huber-White sandwich estimator V = H^{-1} S H^{-1}."
            )
            
            return inference_results
        else:
            return {'error': 'Failed to compute Newey-West standard errors'}
    
    # =============================================================================
    # COMPREHENSIVE MODEL ANALYSIS
    # =============================================================================
    
    def run_full_analysis(self):
        """
        Execute complete model development analysis including estimation,
        validation, and performance assessment
        
        Returns:
        --------
        dict
            Comprehensive analysis results
        """
        
        if self.data is None:
            raise ValueError("Data not loaded. Call load_and_prepare_data() first.")
            
        print(f"MMDA Dynamic Beta Model Analysis - Version {self.model_version}")
        print("=" * 80)
        
        # Extract variables for modeling
        Y = self.data['ILMDHYLD'].values
        FEDL01 = self.data['FEDL01'].values
        FHLK3MSPRD = self.data['FHLK3MSPRD'].values  
        term_spread = self.data['1Y_3M_SPRD'].values
        vol_ratio = self.data['vol_ratio'].values
        rate_change = self.data['FEDL01_change'].fillna(0).values  # For asymmetry test
        
        print(f"\nDataset Summary:")
        print(f"Observations: {len(Y)}")
        print(f"MMDA rate range: {Y.min():.2f}% - {Y.max():.2f}%")
        print(f"Fed funds range: {FEDL01.min():.2f}% - {FEDL01.max():.2f}%")
        print(f"Volatility ratio range: {vol_ratio.min():.2f} - {vol_ratio.max():.2f}")
        
        # Define recent period for performance analysis
        recent_mask = self.data['EOM_Dt'] >= '2022-01-01'
        print(f"Recent period analysis (2022-2025): {recent_mask.sum()} observations")
        
        print("\n1. Conducting Stationarity Analysis...")
        stationarity_results = {}
        for var_name, var_data in [('ILMDHYLD', Y), ('FEDL01', FEDL01)]:
            results = self.stationarity_tests(pd.Series(var_data), var_name)
            stationarity_results.update(results)
            print(f"  {var_name}: ADF p-value = {results[f'{var_name}_adf_pvalue']:.3f}")
        
        # Comprehensive stationarity analysis (levels and first differences)
        print("\n1a. Comprehensive Stationarity Analysis (Levels and First Differences)...")
        series_dict = {
            'MMDA_Rate': Y,
            'Fed_Funds': FEDL01
        }
        comprehensive_stationarity = self.comprehensive_stationarity_analysis(series_dict)
        
        for var_summary in comprehensive_stationarity['summary']:
            print(f"  {var_summary['Variable']}: {var_summary['Conclusion']}")
            print(f"    Level ADF p-value: {var_summary['Level_ADF_pvalue']:.4f}")
            print(f"    First Diff ADF p-value: {var_summary['Diff_ADF_pvalue']:.4f}")
        
        # Cointegration test - run regardless of strict I(1) classification
        # since KPSS may indicate stationarity when ADF does not
        print("\n1b. Cointegration Analysis...")
        mmda_order = comprehensive_stationarity['integration_order'].get('MMDA_Rate', 0)
        ff_order = comprehensive_stationarity['integration_order'].get('Fed_Funds', 0)
        
        # Check KPSS results for first differences as alternative criterion
        mmda_kpss_diff = comprehensive_stationarity['first_differences'].get('MMDA_Rate', {}).get('is_stationary_kpss', False)
        ff_kpss_diff = comprehensive_stationarity['first_differences'].get('Fed_Funds', {}).get('is_stationary_kpss', False)
        
        # Run cointegration test if either:
        # 1. Both series are I(1) by ADF, OR
        # 2. Both first differences are stationary by KPSS (suggesting I(1))
        run_coint = (mmda_order == 1 and ff_order == 1) or (mmda_kpss_diff and ff_kpss_diff)
        
        if run_coint:
            if mmda_order == 1 and ff_order == 1:
                print("  Both series are I(1) by ADF, testing for cointegration...")
            else:
                print("  First differences stationary by KPSS (likely I(1)), testing for cointegration...")
            cointegration_results = self.cointegration_test(Y, FEDL01)
            print(f"  Engle-Granger test statistic: {cointegration_results.get('coint_statistic', np.nan):.4f}")
            print(f"  Cointegration p-value: {cointegration_results.get('coint_pvalue', np.nan):.4f}")
            print(f"  Cointegrated: {'Yes' if cointegration_results.get('is_cointegrated', False) else 'No'}")
            print(f"  Interpretation: {cointegration_results.get('interpretation', 'N/A')}")
        else:
            print(f"  MMDA is I({mmda_order}), Fed Funds is I({ff_order})")
            print("  Cointegration test not applicable")
            cointegration_results = {'note': 'Cointegration test not applicable - series may be I(0) or I(2)'}
        
        print("\n2. Estimating Model Specifications...")
        
        # Enhanced Logistic Model
        print("  - Enhanced logistic model...")
        try:
            result_enhanced = self.estimate_enhanced_logistic(Y, FEDL01, FHLK3MSPRD, term_spread)
            print(f"    Optimization success: {result_enhanced.success}")
            if result_enhanced.success:
                print(f"    Final NLL value: {result_enhanced.fun:.2f}")
        except Exception as e:
            print(f"    Error: {e}")
            result_enhanced = None
        
        # Volatility-Adjusted Model (Recommended)
        print("  - Volatility-adjusted model...")
        try:
            result_vol = self.estimate_volatility_adjusted(Y, FEDL01, FHLK3MSPRD, term_spread, vol_ratio)
            print(f"    Optimization success: {result_vol.success}")
            if result_vol.success:
                print(f"    Final NLL value: {result_vol.fun:.2f}")
        except Exception as e:
            print(f"    Error: {e}")
            result_vol = None
        
        # Quadratic Model (Benchmarking)
        print("  - Quadratic model...")
        try:
            result_quad = self.estimate_quadratic(Y, FEDL01, FHLK3MSPRD, term_spread)
            print(f"    Optimization success: {result_quad.success}")
            if result_quad.success:
                print(f"    Final NLL value: {result_quad.fun:.2f}")
        except Exception as e:
            print(f"    Error: {e}")
            result_quad = None
        
        print("\n3. Generating Predictions and Performance Metrics...")
        
        models = {}
        
        # Process Enhanced Logistic Model
        if result_enhanced and result_enhanced.success:
            params_enh = result_enhanced.x
            beta_enh = self.logistic_beta(FEDL01, params_enh[1], params_enh[2], 
                                        params_enh[3], params_enh[4])
            pred_enhanced = (params_enh[0] + beta_enh * FEDL01 + 
                           params_enh[5] * FHLK3MSPRD + params_enh[6] * term_spread)
            
            metrics_enh = self.calculate_model_metrics(Y, pred_enhanced, 7)
            diagnostics_enh = self.diagnostic_tests(Y - pred_enhanced)
            
            models['enhanced'] = {
                'result': result_enhanced,
                'params': dict(zip(['alpha', 'k', 'm', 'beta_min', 'beta_max', 'gamma_fhlb', 'gamma_term'], 
                                 params_enh)),
                'predictions': pred_enhanced,
                'beta': beta_enh,
                'metrics': metrics_enh,
                'diagnostics': diagnostics_enh,
                'recent_rmse': np.sqrt(np.mean((Y[recent_mask] - pred_enhanced[recent_mask])**2)) if recent_mask.sum() > 0 else np.nan
            }
        
        # Process Volatility-Adjusted Model
        if result_vol and result_vol.success:
            params_vol = result_vol.x
            beta_vol = self.volatility_adjusted_beta(FEDL01, params_vol[1], params_vol[2],
                                                   params_vol[3], params_vol[4], 
                                                   vol_ratio, params_vol[7])
            pred_vol = (params_vol[0] + beta_vol * FEDL01 + 
                       params_vol[5] * FHLK3MSPRD + params_vol[6] * term_spread)
            
            metrics_vol = self.calculate_model_metrics(Y, pred_vol, 8)
            diagnostics_vol = self.diagnostic_tests(Y - pred_vol)
            
            models['vol_adjusted'] = {
                'result': result_vol,
                'params': dict(zip(['alpha', 'k', 'm', 'beta_min', 'beta_max', 'gamma_fhlb', 'gamma_term', 'lambda'], 
                                 params_vol)),
                'predictions': pred_vol,
                'beta': beta_vol,
                'metrics': metrics_vol,
                'diagnostics': diagnostics_vol,
                'recent_rmse': np.sqrt(np.mean((Y[recent_mask] - pred_vol[recent_mask])**2)) if recent_mask.sum() > 0 else np.nan
            }
        
        # Process Quadratic Model
        if result_quad and result_quad.success:
            params_quad = result_quad.x
            beta_quad = self.quadratic_beta(FEDL01, params_quad[1], params_quad[2], params_quad[3],
                                          params_quad[4], params_quad[5])
            pred_quad = (params_quad[0] + beta_quad * FEDL01 + 
                        params_quad[6] * FHLK3MSPRD + params_quad[7] * term_spread)
            
            metrics_quad = self.calculate_model_metrics(Y, pred_quad, 8)
            diagnostics_quad = self.diagnostic_tests(Y - pred_quad)
            
            models['quadratic'] = {
                'result': result_quad,
                'params': dict(zip(['alpha', 'a', 'b', 'c', 'beta_min', 'beta_max', 'gamma_fhlb', 'gamma_term'], 
                                 params_quad)),
                'predictions': pred_quad,
                'beta': beta_quad,
                'metrics': metrics_quad,
                'diagnostics': diagnostics_quad,
                'recent_rmse': np.sqrt(np.mean((Y[recent_mask] - pred_quad[recent_mask])**2)) if recent_mask.sum() > 0 else np.nan
            }
        
        # Print performance summary
        print("\n4. Model Performance Summary:")
        print("-" * 70)
        
        for name, model in models.items():
            metrics = model['metrics']
            print(f"\n{name.upper().replace('_', ' ')} MODEL:")
            print(f"  R²: {metrics['r_squared']:.4f}")
            print(f"  Adjusted R²: {metrics['adj_r_squared']:.4f}")
            print(f"  RMSE: {metrics['rmse']:.4f}%")
            print(f"  AIC: {metrics['aic']:.1f}")
            print(f"  BIC: {metrics['bic']:.1f}")
            print(f"  Recent RMSE (2022-2025): {model['recent_rmse']:.4f}%")
            
            # Diagnostic test results
            diag = model['diagnostics']
            print(f"  Diagnostic Tests:")
            print(f"    Jarque-Bera p-value: {diag['jarque_bera_pvalue']:.3f}")
            print(f"    Breusch-Godfrey p-value: {diag['breusch_godfrey_pvalue']:.3f}")
            print(f"    White's p-value: {diag['white_pvalue']:.3f}")
        
        # Likelihood ratio tests
        if len(models) >= 2:
            print("\n5. Model Comparison Tests:")
            print("-" * 35)
            model_names = list(models.keys())
            for i in range(len(model_names)):
                for j in range(i+1, len(model_names)):
                    name1, name2 = model_names[i], model_names[j]
                    if models[name1]['result'] and models[name2]['result']:
                        ll1 = -models[name1]['result'].fun
                        ll2 = -models[name2]['result'].fun
                        lr_stat = 2 * abs(ll1 - ll2)
                        df = abs(models[name1]['metrics']['params'] - models[name2]['metrics']['params'])
                        print(f"  {name1.replace('_', ' ').title()} vs {name2.replace('_', ' ').title()}:")
                        print(f"    LR Statistic: {lr_stat:.2f} (df={df})")
                        if lr_stat > 3.84:  # Critical value for df=1, α=0.05
                            better_model = name1 if ll1 > ll2 else name2
                            print(f"    Preferred: {better_model.replace('_', ' ').title()}")
                        else:
                            print(f"    No significant difference")
        
        # Store results
        self.models = models
        self.results = {
            'models': models,
            'data': self.data,
            'variables': {
                'Y': Y,
                'FEDL01': FEDL01,
                'FHLK3MSPRD': FHLK3MSPRD,
                'term_spread': term_spread,
                'vol_ratio': vol_ratio,
                'rate_change': rate_change
            },
            'recent_mask': recent_mask,
            'stationarity': stationarity_results,
            'comprehensive_stationarity': comprehensive_stationarity,
            'cointegration': cointegration_results
        }
        
        # Compute robust inference with Newey-West standard errors
        print("\n6. Computing Robust Inference (Newey-West HAC Standard Errors)...")
        robust_inference_results = {}
        for model_name in models.keys():
            try:
                robust_results = self.robust_inference(model_name)
                robust_inference_results[model_name] = robust_results
                print(f"\n  {model_name.upper().replace('_', ' ')} MODEL - Robust Inference:")
                print(f"    Newey-West lags: {robust_results.get('n_lags_used', 'N/A')}")
                for param, info in robust_results.get('parameters', {}).items():
                    print(f"    {param}: estimate={info['estimate']:.4f}, robust SE={info['robust_se']:.4f}, "
                          f"t={info['t_statistic']:.2f}, p={info['p_value']:.4f}")
            except Exception as e:
                print(f"  Warning: Could not compute robust inference for {model_name}: {e}")
                robust_inference_results[model_name] = {'error': str(e)}
        
        self.results['robust_inference'] = robust_inference_results
        
        # Test model residual stationarity (the CORRECT equilibrium test)
        print("\n7. Testing Model Residual Stationarity (Equilibrium Validation)...")
        residual_stationarity_results = self.test_model_residual_stationarity(models)
        
        for model_name, results in residual_stationarity_results.items():
            print(f"\n  {model_name.upper().replace('_', ' ')} MODEL RESIDUALS:")
            if 'error' not in results:
                print(f"    ADF Statistic: {results['adf_statistic']:.4f} (p = {results['adf_pvalue']:.4f})")
                print(f"    KPSS Statistic: {results['kpss_statistic']:.4f} (critical 5% = {results['kpss_critical_5pct']:.4f})")
                print(f"    Conclusion: {results['conclusion']}")
                print(f"    Valid Equilibrium: {'Yes' if results['valid_equilibrium'] else 'No'}")
            else:
                print(f"    Error: {results['error']}")
        
        self.results['residual_stationarity'] = residual_stationarity_results
        
        # Test volatility specification alternatives (Critique #2)
        print("\n8. Testing Volatility Specification Alternatives (Critique #2)...")
        vol_spec_results = self.test_volatility_specifications(
            Y, FEDL01, FHLK3MSPRD, term_spread, vol_ratio, rate_change
        )
        self.results['volatility_specification_tests'] = vol_spec_results
        
        # Test σ* sensitivity (3-year vs 5-year window)
        print("\n9. Testing σ* Calculation Sensitivity (Critique #2)...")
        sigma_sensitivity = self.test_sigma_star_sensitivity(
            Y, FEDL01, FHLK3MSPRD, term_spread
        )
        self.results['sigma_star_sensitivity'] = sigma_sensitivity
        
        # Test regime-specific RMSE (Critique #3)
        print("\n10. Testing Regime-Specific Model Performance (Critique #3)...")
        regime_rmse_results = self.test_regime_specific_rmse(models)
        self.results['regime_specific_rmse'] = regime_rmse_results
        
        return self.results
    
    # =============================================================================
    # VISUALIZATION FUNCTIONS
    # =============================================================================
    
    def create_model_fit_comparison(self, save_path=None, figsize=(16, 10), include_ols=True):
        """
        Create comprehensive model fit comparison visualization
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the figure
        figsize : tuple
            Figure size (width, height)
        include_ols : bool
            Whether to include static OLS benchmark (default True)
        """
        
        if not self.results or 'models' not in self.results:
            print("No results available. Run analysis first.")
            return
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot actual data
        dates = self.data['EOM_Dt']
        actual = self.results['variables']['Y']
        FEDL01 = self.results['variables']['FEDL01']
        
        ax.plot(dates, actual, 'ko', markersize=6, markeredgecolor='white',
                markeredgewidth=1, label='Actual MMDA Rate', alpha=0.8, zorder=5)
        
        # Add Static OLS benchmark
        if include_ols:
            try:
                import statsmodels.api as sm
                X_simple = sm.add_constant(FEDL01)
                ols_simple = sm.OLS(actual, X_simple).fit()
                ols_fitted = ols_simple.fittedvalues
                ols_beta = ols_simple.params[1]
                ols_r2 = ols_simple.rsquared
                ax.plot(dates, ols_fitted, color='#FF8C00', linestyle='--', linewidth=2.5, alpha=0.8,
                        label=f"Static OLS β={ols_beta:.1%} (R²={ols_r2:.3f})")
            except Exception as e:
                print(f"Could not add OLS benchmark: {e}")
        
        # Plot model predictions
        colors = {'enhanced': '#E74C3C', 'vol_adjusted': '#3498DB', 'quadratic': '#2ECC71'}
        styles = {'enhanced': '-', 'vol_adjusted': '-', 'quadratic': '--'}
        widths = {'enhanced': 2.0, 'vol_adjusted': 3.0, 'quadratic': 2.0}
        
        for name, model in self.results['models'].items():
            if name in colors:
                metrics = model['metrics']
                ax.plot(dates, model['predictions'],
                       color=colors[name], linestyle=styles[name], 
                       linewidth=widths[name], alpha=0.8,
                       label=f"{name.replace('_', ' ').title()} (R²={metrics['r_squared']:.3f})")
        
        # Highlight 2022-2025 period
        if self.results['recent_mask'].sum() > 0:
            recent_start = dates[self.results['recent_mask']].iloc[0]
            recent_end = dates.iloc[-1]
            ax.axvspan(recent_start, recent_end, alpha=0.15, color='gold',
                      label='2022-2025 Focus Period', zorder=1)
        
        # Formatting
        ax.set_xlabel('Date', fontsize=14, fontweight='bold')
        ax.set_ylabel('MMDA Rate (%)', fontsize=14, fontweight='bold')
        ax.set_title('MMDA Model Fit Comparison: Static vs Dynamic Beta (2017-2025)',
                    fontsize=16, fontweight='bold', pad=20)
        ax.legend(fontsize=11, loc='upper left', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        # Add performance statistics (positioned in bottom right to avoid legend overlap)
        if len(self.results['models']) > 0:
            textstr = "2022-2025 RMSE:\n"
            if include_ols:
                recent_mask = self.results['recent_mask']
                ols_recent_rmse = np.sqrt(np.mean((actual[recent_mask] - ols_fitted[recent_mask])**2))
                textstr += f"• Static OLS: {ols_recent_rmse:.3f}%\n"
            for name, model in self.results['models'].items():
                if name in colors:
                    textstr += f"• {name.replace('_', ' ').title()}: {model['recent_rmse']:.3f}%\n"
            
            textstr = textstr.rstrip('\n')
            props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray')
            ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=10,
                   verticalalignment='bottom', horizontalalignment='right', 
                   bbox=props, family='monospace')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Model comparison chart saved to {save_path}")
            plt.close()
    
    def create_beta_evolution_chart(self, save_path=None, figsize=(14, 9)):
        """
        Create dynamic beta evolution comparison across rate environments
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the figure
        figsize : tuple
            Figure size (width, height)
        """
        
        if not self.results or 'models' not in self.results:
            print("No results available. Run analysis first.")
            return
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create rate grid for comparison
        rate_grid = np.linspace(0, 6, 100)
        vol_ratio_mean = np.mean(self.results['variables']['vol_ratio'])
        
        colors = {'enhanced': '#E74C3C', 'vol_adjusted': '#3498DB', 'quadratic': '#2ECC71'}
        styles = {'enhanced': '-', 'vol_adjusted': '-', 'quadratic': '--'}
        markers = {'enhanced': 'o', 'vol_adjusted': 's', 'quadratic': '^'}
        
        # Plot beta evolution for each model
        for name, model in self.results['models'].items():
            if name in colors:
                params = model['params']
                
                if name == 'enhanced':
                    beta_grid = self.logistic_beta(rate_grid, params['k'], params['m'],
                                                 params['beta_min'], params['beta_max'])
                    inflection = params['m']
                elif name == 'vol_adjusted':
                    beta_grid = self.volatility_adjusted_beta(rate_grid, params['k'], params['m'],
                                                            params['beta_min'], params['beta_max'],
                                                            vol_ratio_mean, params['lambda'])
                    inflection = params['m']
                elif name == 'quadratic':
                    beta_grid = self.quadratic_beta(rate_grid, params['a'], params['b'], params['c'],
                                                  params['beta_min'], params['beta_max'])
                    inflection = None
                
                ax.plot(rate_grid, beta_grid,
                       color=colors[name], linestyle=styles[name], linewidth=3,
                       marker=markers[name], markersize=8, markevery=15,
                       markeredgecolor='white', markeredgewidth=1, alpha=0.8,
                       label=f"{name.replace('_', ' ').title()} Model")
                
                # Mark inflection points
                if inflection is not None:
                    if name == 'enhanced':
                        inflection_beta = self.logistic_beta(np.array([inflection]), params['k'], params['m'],
                                                           params['beta_min'], params['beta_max'])
                    else:
                        inflection_beta = self.volatility_adjusted_beta(np.array([inflection]), params['k'], params['m'],
                                                                      params['beta_min'], params['beta_max'],
                                                                      vol_ratio_mean, params['lambda'])
                    ax.scatter([inflection], [inflection_beta], color=colors[name], s=120,
                             marker='D', edgecolor='white', linewidth=2, zorder=6,
                             label=f"{name.replace('_', ' ').title()} Inflection ({inflection:.1f}%)")
        
        # Reference lines
        for level in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
            ax.axhline(y=level, color='gray', linestyle=':', alpha=0.4, linewidth=1)
            ax.text(5.8, level, f'{level:.0%}', fontsize=9, color='gray',
                   verticalalignment='center')
        
        # Formatting
        ax.set_xlabel('Federal Funds Rate (%)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Dynamic Beta (Deposit Rate Sensitivity)', fontsize=14, fontweight='bold')
        ax.set_title('Dynamic Beta Evolution Across Interest Rate Environments',
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlim(0, 6)
        ax.set_ylim(0.30, 0.65)
        ax.legend(fontsize=11, loc='center right', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Beta evolution chart saved to {save_path}")
            plt.close()
    
    def create_asymmetric_beta_evolution_chart(self, save_path=None, figsize=(16, 10)):
        """
        Create asymmetric dynamic beta evolution with separate panels for rising vs falling rates.
        
        This is Figure 3 in the paper - shows how the beta curve differs based on rate direction.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the figure
        figsize : tuple
            Figure size (width, height)
        """
        
        if 'volatility_specification_tests' not in self.results:
            print("Volatility specification tests not available. Run full analysis first.")
            return
        
        vol_spec = self.results.get('volatility_specification_tests', {})
        asym_model = vol_spec.get('models', {}).get('asymmetric', {})
        
        if not asym_model.get('success', False):
            print("Asymmetric model estimation not available.")
            return
        
        # Get asymmetric parameters
        lambda_up = asym_model.get('lambda_up', 0.227)
        lambda_down = asym_model.get('lambda_down', 0.188)
        
        # Get symmetric parameters for comparison  
        sym_model = vol_spec.get('models', {}).get('symmetric', {})
        lambda_sym = sym_model.get('lambda', 0.224) if sym_model.get('success') else 0.224
        
        # Get base parameters from vol_adjusted model
        if 'vol_adjusted' in self.results.get('models', {}):
            base_params = self.results['models']['vol_adjusted']['params']
            k = base_params['k']
            m = base_params['m']
            beta_min = base_params['beta_min']
            beta_max = base_params['beta_max']
        else:
            # Fallback defaults
            k, m, beta_min, beta_max = 0.586, 2.99, 0.40, 0.70
        
        # Create figure with two panels
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Create rate grid
        rate_grid = np.linspace(0, 6, 100)
        
        # Volatility ratio scenarios (low, average, high)
        vol_scenarios = {
            'Low Volatility (σ/σ* = 0.5)': 0.5,
            'Average Volatility (σ/σ* = 1.0)': 1.0,
            'High Volatility (σ/σ* = 2.0)': 2.0
        }
        
        colors = {'Low Volatility (σ/σ* = 0.5)': '#2ECC71', 
                  'Average Volatility (σ/σ* = 1.0)': '#3498DB',
                  'High Volatility (σ/σ* = 2.0)': '#E74C3C'}
        
        linestyles = {'Low Volatility (σ/σ* = 0.5)': '-',
                      'Average Volatility (σ/σ* = 1.0)': '-',
                      'High Volatility (σ/σ* = 2.0)': '--'}
        
        # Panel A: Rising Rate Environment (λ_up = 0.227)
        ax1 = axes[0]
        ax1.set_title('Panel A: Rising Rate Environment\n(Higher Volatility Dampening: λ_up = {:.3f})'.format(lambda_up),
                     fontsize=13, fontweight='bold', pad=15)
        
        for scenario_name, vol_ratio in vol_scenarios.items():
            beta_grid = self.volatility_adjusted_beta(
                rate_grid, k, m, beta_min, beta_max, vol_ratio, lambda_up
            )
            ax1.plot(rate_grid, beta_grid, 
                    color=colors[scenario_name], 
                    linestyle=linestyles[scenario_name],
                    linewidth=2.5, label=scenario_name, alpha=0.85)
        
        # Mark inflection point
        inflection_beta_up = self.volatility_adjusted_beta(
            np.array([m]), k, m, beta_min, beta_max, 1.0, lambda_up
        )
        ax1.scatter([m], inflection_beta_up, color='#9B59B6', s=150, marker='D',
                   edgecolor='white', linewidth=2, zorder=6, 
                   label=f'Inflection Point ({m:.1f}%)')
        
        # Add shading to show dampening effect
        beta_no_dampen = self.logistic_beta(rate_grid, k, m, beta_min, beta_max)
        beta_high_vol = self.volatility_adjusted_beta(rate_grid, k, m, beta_min, beta_max, 2.0, lambda_up)
        ax1.fill_between(rate_grid, beta_high_vol, beta_no_dampen, 
                        alpha=0.15, color='#E74C3C', 
                        label='Dampening Effect (High Vol)')
        
        ax1.set_xlabel('Federal Funds Rate (%)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Dynamic Beta (Deposit Rate Sensitivity)', fontsize=12, fontweight='bold')
        ax1.set_xlim(0, 6)
        ax1.set_ylim(0.30, 0.75)
        ax1.legend(fontsize=9, loc='lower right', framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        
        # Add annotation
        ax1.annotate('Banks delay repricing\nmore aggressively\nduring rate hikes',
                    xy=(4.5, 0.45), fontsize=10, style='italic',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Panel B: Falling Rate Environment (λ_down = 0.188)
        ax2 = axes[1]
        ax2.set_title('Panel B: Falling Rate Environment\n(Lower Volatility Dampening: λ_down = {:.3f})'.format(lambda_down),
                     fontsize=13, fontweight='bold', pad=15)
        
        for scenario_name, vol_ratio in vol_scenarios.items():
            beta_grid = self.volatility_adjusted_beta(
                rate_grid, k, m, beta_min, beta_max, vol_ratio, lambda_down
            )
            ax2.plot(rate_grid, beta_grid,
                    color=colors[scenario_name],
                    linestyle=linestyles[scenario_name],
                    linewidth=2.5, label=scenario_name, alpha=0.85)
        
        # Mark inflection point
        inflection_beta_down = self.volatility_adjusted_beta(
            np.array([m]), k, m, beta_min, beta_max, 1.0, lambda_down
        )
        ax2.scatter([m], inflection_beta_down, color='#9B59B6', s=150, marker='D',
                   edgecolor='white', linewidth=2, zorder=6,
                   label=f'Inflection Point ({m:.1f}%)')
        
        # Add shading to show dampening effect
        beta_high_vol_down = self.volatility_adjusted_beta(rate_grid, k, m, beta_min, beta_max, 2.0, lambda_down)
        ax2.fill_between(rate_grid, beta_high_vol_down, beta_no_dampen,
                        alpha=0.15, color='#E74C3C',
                        label='Dampening Effect (High Vol)')
        
        ax2.set_xlabel('Federal Funds Rate (%)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Dynamic Beta (Deposit Rate Sensitivity)', fontsize=12, fontweight='bold')
        ax2.set_xlim(0, 6)
        ax2.set_ylim(0.30, 0.75)
        ax2.legend(fontsize=9, loc='lower right', framealpha=0.9)
        ax2.grid(True, alpha=0.3)
        
        # Add annotation
        ax2.annotate('Banks pass through\nmore quickly when\nrates are falling',
                    xy=(4.5, 0.50), fontsize=10, style='italic',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        
        # Add overall title
        fig.suptitle('Figure 3: Asymmetric Volatility Dampening in MMDA Repricing\n'
                    'State-Dependent Deposit Rate Sensitivity Across Rate Environments',
                    fontsize=14, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Asymmetric beta evolution chart saved to {save_path}")
            plt.close()
        
        return fig

    def create_residual_analysis(self, model_name='vol_adjusted', save_path=None, figsize=(16, 12)):
        """
        Create comprehensive residual analysis for specified model
        
        Parameters:
        -----------
        model_name : str
            Name of model to analyze ('vol_adjusted', 'enhanced', 'quadratic')
        save_path : str, optional
            Path to save the figure
        figsize : tuple
            Figure size (width, height)
        """
        
        if not self.results or model_name not in self.results['models']:
            print(f"Model '{model_name}' not available for analysis.")
            return
            
        model = self.results['models'][model_name]
        Y = self.results['variables']['Y']
        residuals = Y - model['predictions']
        dates = self.data['EOM_Dt']
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'Residual Analysis: {model_name.replace("_", " ").title()} Model',
                    fontsize=16, fontweight='bold', y=0.95)
        
        # 1. Residuals vs Fitted
        axes[0, 0].scatter(model['predictions'], residuals, alpha=0.7, s=50,
                          color='steelblue', edgecolor='white', linewidth=0.5)
        axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[0, 0].set_xlabel('Fitted Values (%)', fontweight='bold')
        axes[0, 0].set_ylabel('Residuals (%)', fontweight='bold')
        axes[0, 0].set_title('Residuals vs. Fitted Values', fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Q-Q Plot
        stats.probplot(residuals, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot: Normal Distribution', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Residuals over time
        axes[1, 0].plot(dates, residuals, 'steelblue', alpha=0.8, linewidth=1.5)
        axes[1, 0].scatter(dates, residuals, alpha=0.7, s=30, color='steelblue',
                          edgecolor='white', linewidth=0.5)
        axes[1, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[1, 0].set_xlabel('Date', fontweight='bold')
        axes[1, 0].set_ylabel('Residuals (%)', fontweight='bold')
        axes[1, 0].set_title('Residuals Over Time', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Highlight recent period
        if self.results['recent_mask'].sum() > 0:
            recent_start = dates[self.results['recent_mask']].iloc[0]
            recent_end = dates.iloc[-1]
            axes[1, 0].axvspan(recent_start, recent_end, alpha=0.2, color='gold',
                              label='2022-2025 Period')
            axes[1, 0].legend()
        
        # 4. Histogram with normal overlay
        n_bins = 25
        axes[1, 1].hist(residuals, bins=n_bins, density=True, alpha=0.7,
                       color='steelblue', edgecolor='white', linewidth=0.5)
        
        # Normal distribution overlay
        mu, sigma = np.mean(residuals), np.std(residuals)
        x = np.linspace(residuals.min(), residuals.max(), 100)
        normal_overlay = stats.norm.pdf(x, mu, sigma)
        axes[1, 1].plot(x, normal_overlay, 'red', linewidth=3, alpha=0.8,
                       label=f'Normal(μ={mu:.4f}, σ={sigma:.4f})')
        axes[1, 1].set_xlabel('Residuals (%)', fontweight='bold')
        axes[1, 1].set_ylabel('Density', fontweight='bold')
        axes[1, 1].set_title('Distribution of Residuals', fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92, bottom=0.08)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Residual analysis chart saved to {save_path}")
            plt.close()
    
    def create_data_visualization_dashboard(self, save_path=None, figsize=(20, 12)):
        """
        Create comprehensive data visualization dashboard
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the figure
        figsize : tuple
            Figure size (width, height)
        """
        
        if self.data is None:
            print("No data available. Load data first.")
            return
            
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        dates = self.data['EOM_Dt']
        
        # 1. Time series of key rates
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(dates, self.data['FEDL01'], 'navy', linewidth=2, label='Fed Funds Rate')
        ax1.plot(dates, self.data['ILMDHYLD'], 'darkred', linewidth=2, label='MMDA Rate')
        ax1.set_ylabel('Rate (%)', fontweight='bold')
        ax1.set_title('Interest Rate Evolution (2017-2025)', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Volatility evolution
        ax2 = fig.add_subplot(gs[0, 2:])
        ax2.plot(dates, self.data['vol_24m'], 'goldenrod', linewidth=2)
        ax2.fill_between(dates, self.data['vol_24m'], alpha=0.3, color='goldenrod')
        ax2.set_ylabel('Volatility (%)', fontweight='bold')
        ax2.set_title('Fed Funds Rate Volatility (24-Month Rolling)', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. Spread analysis
        ax3 = fig.add_subplot(gs[1, :2])
        ax3.plot(dates, self.data['FHLK3MSPRD'], 'purple', linewidth=2, label='FHLB Spread')
        if '1Y_3M_SPRD' in self.data.columns:
            ax3.plot(dates, self.data['1Y_3M_SPRD'], 'orange', linewidth=2, label='Term Spread')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.set_ylabel('Spread (bp)', fontweight='bold')
        ax3.set_title('Market Spreads Evolution', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Rate scatter plot
        ax4 = fig.add_subplot(gs[1, 2:])
        scatter = ax4.scatter(self.data['FEDL01'], self.data['ILMDHYLD'], 
                            c=self.data.index, cmap='viridis', alpha=0.7, s=50)
        ax4.plot([0, 6], [0, 6], 'r--', alpha=0.5, label='45° Line')
        ax4.set_xlabel('Fed Funds Rate (%)', fontweight='bold')
        ax4.set_ylabel('MMDA Rate (%)', fontweight='bold')
        ax4.set_title('Rate Relationship Over Time', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax4, label='Time Period')
        
        # 5. Distribution analysis
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.hist(self.data['FEDL01'], bins=20, alpha=0.7, color='navy', density=True)
        ax5.set_xlabel('Fed Funds Rate (%)', fontweight='bold')
        ax5.set_ylabel('Density', fontweight='bold')
        ax5.set_title('Fed Funds Distribution', fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.hist(self.data['ILMDHYLD'], bins=20, alpha=0.7, color='darkred', density=True)
        ax6.set_xlabel('MMDA Rate (%)', fontweight='bold')
        ax6.set_ylabel('Density', fontweight='bold')
        ax6.set_title('MMDA Rate Distribution', fontweight='bold')
        ax6.grid(True, alpha=0.3)
        
        # 6. Correlation heatmap
        ax7 = fig.add_subplot(gs[2, 2])
        corr_vars = ['FEDL01', 'ILMDHYLD', 'FHLK3MSPRD', 'vol_24m']
        corr_data = self.data[corr_vars].corr()
        im = ax7.imshow(corr_data, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        ax7.set_xticks(range(len(corr_vars)))
        ax7.set_yticks(range(len(corr_vars)))
        ax7.set_xticklabels([v.replace('_', ' ') for v in corr_vars], rotation=45)
        ax7.set_yticklabels([v.replace('_', ' ') for v in corr_vars])
        ax7.set_title('Variable Correlations', fontweight='bold')
        
        # Add correlation values
        for i in range(len(corr_vars)):
            for j in range(len(corr_vars)):
                ax7.text(j, i, f'{corr_data.iloc[i, j]:.2f}', ha='center', va='center',
                        color='white' if abs(corr_data.iloc[i, j]) > 0.5 else 'black')
        
        # 7. Summary statistics
        ax8 = fig.add_subplot(gs[2, 3])
        ax8.axis('off')
        
        stats_text = "Dataset Summary Statistics:\n\n"
        stats_text += f"Observations: {len(self.data)}\n"
        stats_text += f"Time Period: {dates.min().strftime('%Y-%m')} to {dates.max().strftime('%Y-%m')}\n\n"
        stats_text += f"Fed Funds Rate:\n"
        stats_text += f"  Mean: {self.data['FEDL01'].mean():.2f}%\n"
        stats_text += f"  Std: {self.data['FEDL01'].std():.2f}%\n"
        stats_text += f"  Range: {self.data['FEDL01'].min():.2f}% - {self.data['FEDL01'].max():.2f}%\n\n"
        stats_text += f"MMDA Rate:\n"
        stats_text += f"  Mean: {self.data['ILMDHYLD'].mean():.2f}%\n"
        stats_text += f"  Std: {self.data['ILMDHYLD'].std():.2f}%\n"
        stats_text += f"  Range: {self.data['ILMDHYLD'].min():.2f}% - {self.data['ILMDHYLD'].max():.2f}%\n\n"
        stats_text += f"Correlation (Fed-MMDA): {self.data[['FEDL01', 'ILMDHYLD']].corr().iloc[0,1]:.3f}"
        
        ax8.text(0.05, 0.95, stats_text, transform=ax8.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.suptitle('MMDA Dynamic Beta Model - Data Analysis Dashboard', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Data dashboard saved to {save_path}")
            plt.close()
    
    def generate_model_report(self):
        """
        Generate comprehensive text report of model results
        
        Returns:
        --------
        str
            Formatted model report
        """
        
        if not self.results:
            return "No results available. Run analysis first."
            
        report = []
        report.append("=" * 80)
        report.append("MMDA DYNAMIC BETA MODEL - COMPREHENSIVE ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Model Version: {self.model_version}")
        report.append(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Dataset summary
        report.append("DATASET SUMMARY:")
        report.append("-" * 20)
        report.append(f"Observations: {len(self.data)}")
        report.append(f"Time Period: {self.data['EOM_Dt'].min().strftime('%Y-%m-%d')} to {self.data['EOM_Dt'].max().strftime('%Y-%m-%d')}")
        report.append(f"Fed Funds Range: {self.results['variables']['FEDL01'].min():.2f}% - {self.results['variables']['FEDL01'].max():.2f}%")
        report.append(f"MMDA Rate Range: {self.results['variables']['Y'].min():.2f}% - {self.results['variables']['Y'].max():.2f}%")
        report.append("")
        
        # Model performance comparison
        report.append("MODEL PERFORMANCE COMPARISON:")
        report.append("-" * 35)
        report.append(f"{'Model':<20} {'R²':<8} {'RMSE':<8} {'AIC':<10} {'Recent RMSE':<12}")
        report.append("-" * 70)
        
        for name, model in self.results['models'].items():
            metrics = model['metrics']
            report.append(f"{name.replace('_', ' ').title():<20} "
                         f"{metrics['r_squared']:.4f}   "
                         f"{metrics['rmse']:.4f}   "
                         f"{metrics['aic']:.1f}      "
                         f"{model['recent_rmse']:.4f}")
        
        report.append("")
        
        # Recommended model parameters
        if 'vol_adjusted' in self.results['models']:
            model = self.results['models']['vol_adjusted']
            params = model['params']
            report.append("RECOMMENDED MODEL PARAMETERS (Volatility-Adjusted):")
            report.append("-" * 55)
            for param_name, param_value in params.items():
                if param_name == 'm':
                    report.append(f"{param_name.ljust(15)}: {param_value:.4f} (Inflection at {param_value:.1f}% Fed Funds)")
                elif param_name in ['beta_min', 'beta_max']:
                    report.append(f"{param_name.ljust(15)}: {param_value:.4f} ({param_value:.1%})")
                else:
                    report.append(f"{param_name.ljust(15)}: {param_value:.4f}")
            report.append("")
            
            # Diagnostic tests
            diag = model['diagnostics']
            report.append("DIAGNOSTIC TEST RESULTS:")
            report.append("-" * 25)
            report.append(f"Jarque-Bera (normality) p-value: {diag['jarque_bera_pvalue']:.3f}")
            report.append(f"Breusch-Godfrey (autocorr) p-value: {diag['breusch_godfrey_pvalue']:.3f}")
            report.append(f"White's (heteroscedasticity) p-value: {diag['white_pvalue']:.3f}")
            
            # Interpretation
            report.append("")
            report.append("ECONOMIC INTERPRETATION:")
            report.append("-" * 25)
            report.append(f"• Competition intensifies above {params['m']:.1f}% Fed Funds rate")
            report.append(f"• Beta ranges from {params['beta_min']:.1%} to {params['beta_max']:.1%}")
            report.append(f"• Volatility dampening factor: {params['lambda']:.1%}")
            report.append(f"• Model explains {model['metrics']['r_squared']:.1%} of deposit rate variation")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)

# =============================================================================
# MAIN EXECUTION AND EXAMPLE USAGE
# =============================================================================

def main():
    """
    Main execution function demonstrating complete model workflow
    """
    
    print("MMDA Dynamic Beta Model - Complete Implementation")
    print("=" * 60)
    print("Initializing model development framework...")
    
    # Initialize model
    model = MMDADynamicBetaModel(model_version="1.0")
    
    # Load and prepare data (replace with actual file path)
    print("\nStep 1: Loading and preparing data...")
    data_file = "bankratemma.csv"  # Update this path as needed
    
    try:
        data = model.load_and_prepare_data(data_file)
        if data is None:
            print("Failed to load data. Please check file path and format.")
            return
    except FileNotFoundError:
        print(f"Data file '{data_file}' not found.")
        print("Please ensure the CSV file is in the current directory or update the path.")
        return
    
    # Run comprehensive analysis
    print("\nStep 2: Running comprehensive model analysis...")
    results = model.run_full_analysis()
    
    # Generate visualizations
    print("\nStep 3: Creating comprehensive visualizations...")
    
    try:
        # Data dashboard
        print("  - Creating data analysis dashboard...")
        model.create_data_visualization_dashboard(save_path='mmda_data_dashboard.png')
        
        # Model fit comparison
        print("  - Creating model fit comparison...")
        model.create_model_fit_comparison(save_path='mmda_model_fit_comparison.png')
        
        # Beta evolution
        print("  - Creating beta evolution analysis...")
        model.create_beta_evolution_chart(save_path='mmda_beta_evolution.png')
        
        # Residual analysis
        print("  - Creating residual analysis...")
        model.create_residual_analysis(model_name='vol_adjusted', 
                                     save_path='mmda_residual_analysis.png')
        
        print("All visualizations created successfully!")
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")
    
    # Generate and display comprehensive report
    print("\nStep 4: Generating comprehensive model report...")
    report = model.generate_model_report()
    print(report)
    
    # Save report to file
    with open('mmda_model_analysis_report.txt', 'w') as f:
        f.write(report)
    print("\nDetailed report saved to 'mmda_model_analysis_report.txt'")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETED SUCCESSFULLY!")
    print("All results, visualizations, and reports have been generated.")
    print("Model is ready for MRM review and implementation.")
    
    return model

if __name__ == "__main__":
    # Execute main analysis
    mmda_model = main()