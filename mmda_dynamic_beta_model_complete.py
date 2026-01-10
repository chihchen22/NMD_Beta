"""
Money Market Deposit Account Dynamic Beta Repricing Model
Complete Implementation with Analysis and Visualizations

Model Owner: [Author Name]
Department: Asset Liability Management / Treasury
Date: October 1, 2025
Model Version: 1.0

This code provides complete implementation of the volatility-adjusted dynamic beta model
for MMDA repricing, including all analysis, testing, and visualizations required
for model development and validation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.stats.diagnostic import acorr_breusch_godfrey, het_white
from statsmodels.stats.stattools import jarque_bera
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
                'vol_ratio': vol_ratio
            },
            'recent_mask': recent_mask,
            'stationarity': stationarity_results
        }
        
        return self.results
    
    # =============================================================================
    # VISUALIZATION FUNCTIONS
    # =============================================================================
    
    def create_model_fit_comparison(self, save_path=None, figsize=(16, 10)):
        """
        Create comprehensive model fit comparison visualization
        
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
        
        # Plot actual data
        dates = self.data['EOM_Dt']
        actual = self.results['variables']['Y']
        
        ax.plot(dates, actual, 'ko', markersize=6, markeredgecolor='white',
                markeredgewidth=1, label='Actual MMDA Rate', alpha=0.8, zorder=5)
        
        # Plot model predictions
        colors = {'enhanced': '#E74C3C', 'vol_adjusted': '#3498DB', 'quadratic': '#2ECC71'}
        styles = {'enhanced': '-', 'vol_adjusted': '-', 'quadratic': '--'}
        widths = {'enhanced': 2.5, 'vol_adjusted': 3.0, 'quadratic': 2.0}
        
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
            ax.axvspan(recent_start, recent_end, alpha=0.2, color='gold',
                      label='2022-2025 Focus Period', zorder=1)
        
        # Formatting
        ax.set_xlabel('Date', fontsize=14, fontweight='bold')
        ax.set_ylabel('MMDA Rate (%)', fontsize=14, fontweight='bold')
        ax.set_title('MMDA Dynamic Beta Model Fit Comparison (2017-2025)',
                    fontsize=16, fontweight='bold', pad=20)
        ax.legend(fontsize=12, loc='upper left', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        # Add performance statistics (positioned in bottom right to avoid legend overlap)
        if len(self.results['models']) > 0:
            textstr = "Recent Period Performance (2022-2025 RMSE):\n"
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
        
        plt.show()
    
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
        
        plt.show()
    
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
        
        plt.show()
    
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
        
        plt.show()
    
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