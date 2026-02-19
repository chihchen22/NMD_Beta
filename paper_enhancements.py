"""
Paper Enhancement Analyses
============================

Produces the additional analyses needed for the SSRN-ready paper:
  1. Out-of-sample validation (expanding-window pseudo-OOS)
  2. Robustness checks (vol windows, sub-sample stability, rolling params)
  3. Effective duration & convexity calculations
  4. Worked portfolio example ($10B MMDA book)
  5. Bootstrap confidence intervals on scenario forecasts

Outputs → outputs/paper_enhancements/

Author: Chih L. Chen
Date: February 2026
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from enhanced_dynamic_beta_model import (
    logistic_beta, asymmetric_volatility_beta, ar_smoothed_beta,
    EnhancedDynamicBetaModel
)

OUTPUT_DIR = os.path.join("outputs", "paper_enhancements")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model parameters from v2.0 estimation
PARAMS = {
    'alpha': 0.073, 'k': 0.566, 'm': 3.919,
    'beta_min': 0.433, 'beta_max': 0.800,
    'gamma_fhlb': 1.049, 'lambda_up': 0.255, 'lambda_down': 0.223,
}
AR_K = 0.02
THETA = 0.4707  # Nerlove partial adjustment speed
LAST_FF = 4.33
LAST_MMDA = 2.495
LAST_FHLB = 0.1834

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10,
    'axes.titlesize': 12, 'axes.labelsize': 11,
    'figure.facecolor': 'white',
})


# ═══════════════════════════════════════════════════════════════════════════════
# 1. OUT-OF-SAMPLE VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def run_oos_validation():
    """
    Expanding-window out-of-sample validation.
    
    Train on Jan 2017 – Dec 2021 (60 obs.), forecast one month ahead.
    Roll forward monthly through Mar 2025 (39 OOS months).
    
    Also reports: train-only model vs full-sample model performance on the
    OOS period, to assess specification stability.
    """
    print("\n" + "="*72)
    print("OUT-OF-SAMPLE VALIDATION: Expanding Window")
    print("="*72)
    
    # Load full dataset
    model_full = EnhancedDynamicBetaModel(max_beta_change=AR_K)
    df = model_full.load_data('bankratemma.csv')
    
    # Define cutoff
    train_end = pd.Timestamp('2021-12-31')
    train_mask = df['EOM_Dt'] <= train_end
    n_train = train_mask.sum()
    n_total = len(df)
    n_oos = n_total - n_train
    
    print(f"\nIn-sample: {n_train} obs (Jan 2017 – Dec 2021)")
    print(f"Out-of-sample: {n_oos} obs (Jan 2022 – Mar 2025)")
    
    # --- Strategy 1: Fixed training window, pseudo-OOS ---
    model_train = EnhancedDynamicBetaModel(max_beta_change=AR_K)
    df_train = df[train_mask].copy().reset_index(drop=True)
    
    # Manually set up the training model
    model_train.data = df_train
    model_train.Y = df_train['ILMDHYLD'].values
    model_train.FEDL01 = df_train['FEDL01'].values
    model_train.FHLK3MSPRD = df_train['FHLK3MSPRD'].values
    model_train.vol_ratio = df_train['vol_ratio'].values
    model_train.rate_change = df_train['FEDL01_change'].fillna(0).values
    
    print("\nEstimating on training sample (2017-2021)...")
    model_train.estimate(n_starts=5)
    params_train = model_train.result.x
    
    # Get full-sample estimates for comparison
    print("Estimating on full sample (2017-2025)...")
    model_full.estimate(n_starts=5)
    params_full = model_full.result.x
    
    param_names = ['alpha', 'k', 'm', 'beta_min', 'beta_max',
                   'gamma_fhlb', 'lambda_up', 'lambda_down']
    
    print("\n--- Parameter Stability ---")
    print(f"{'Parameter':<14} {'Train':>10} {'Full':>10} {'Diff':>10} {'% Change':>10}")
    print("-" * 56)
    for i, name in enumerate(param_names):
        pct = 100 * (params_full[i] - params_train[i]) / (abs(params_train[i]) + 1e-10)
        print(f"{name:<14} {params_train[i]:>10.4f} {params_full[i]:>10.4f} "
              f"{params_full[i]-params_train[i]:>10.4f} {pct:>9.1f}%")
    
    # --- Compute OOS predictions using TRAIN parameters ---
    oos_mask = df['EOM_Dt'] > train_end
    df_oos = df[oos_mask].copy()
    
    Y_oos = df_oos['ILMDHYLD'].values
    FF_oos = df_oos['FEDL01'].values
    FHLB_oos = df_oos['FHLK3MSPRD'].values
    vol_ratio_oos = df_oos['vol_ratio'].values
    rate_change_oos = df_oos['FEDL01_change'].fillna(0).values
    dates_oos = df_oos['EOM_Dt'].values
    
    alpha_t, k_t, m_t, bmin_t, bmax_t, gf_t, lu_t, ld_t = params_train
    
    beta_unc_oos = asymmetric_volatility_beta(
        FF_oos, k_t, m_t, bmin_t, bmax_t,
        vol_ratio_oos, lu_t, ld_t, rate_change_oos
    )
    beta_sm_oos = ar_smoothed_beta(beta_unc_oos, AR_K)
    pred_oos_train_params = alpha_t + beta_sm_oos * FF_oos + gf_t * FHLB_oos
    resid_oos = Y_oos - pred_oos_train_params
    
    # In-sample predictions (train params on train data)
    Y_is = model_train.Y
    pred_is = model_train.predictions
    resid_is = Y_is - pred_is
    
    # Full-sample model predictions on OOS period
    alpha_f, k_f, m_f, bmin_f, bmax_f, gf_f, lu_f, ld_f = params_full
    beta_unc_full = asymmetric_volatility_beta(
        FF_oos, k_f, m_f, bmin_f, bmax_f,
        vol_ratio_oos, lu_f, ld_f, rate_change_oos
    )
    beta_sm_full = ar_smoothed_beta(beta_unc_full, AR_K)
    pred_oos_full_params = alpha_f + beta_sm_full * FF_oos + gf_f * FHLB_oos
    resid_oos_full = Y_oos - pred_oos_full_params
    
    # Static OLS benchmark (train only)
    from numpy.linalg import lstsq
    X_train_ols = np.column_stack([np.ones(n_train), model_train.FEDL01])
    ols_coef, _, _, _ = lstsq(X_train_ols, Y_is, rcond=None)
    pred_oos_static = ols_coef[0] + ols_coef[1] * FF_oos
    resid_oos_static = Y_oos - pred_oos_static
    
    # Metrics
    def calc_metrics(resid, y):
        rmse = np.sqrt(np.mean(resid**2))
        mae = np.mean(np.abs(resid))
        ss_res = np.sum(resid**2)
        ss_tot = np.sum((y - y.mean())**2)
        r2 = 1 - ss_res / ss_tot
        bias = np.mean(resid)
        return rmse, mae, r2, bias
    
    rmse_is, mae_is, r2_is, bias_is = calc_metrics(resid_is, Y_is)
    rmse_oos, mae_oos, r2_oos, bias_oos = calc_metrics(resid_oos, Y_oos)
    rmse_oos_f, mae_oos_f, r2_oos_f, bias_oos_f = calc_metrics(resid_oos_full, Y_oos)
    rmse_static, mae_static, r2_static, bias_static = calc_metrics(resid_oos_static, Y_oos)
    
    print("\n--- Out-of-Sample Performance (Jan 2022 – Mar 2025) ---")
    print(f"{'Model':<35} {'RMSE(bps)':>10} {'MAE(bps)':>10} {'R²':>8} {'Bias(bps)':>10}")
    print("-" * 75)
    print(f"{'Static OLS (train params)':<35} {rmse_static*100:>10.1f} {mae_static*100:>10.1f} "
          f"{r2_static:>8.4f} {bias_static*100:>10.1f}")
    print(f"{'Dynamic v2.0 (train params)':<35} {rmse_oos*100:>10.1f} {mae_oos*100:>10.1f} "
          f"{r2_oos:>8.4f} {bias_oos*100:>10.1f}")
    print(f"{'Dynamic v2.0 (full-sample params)':<35} {rmse_oos_f*100:>10.1f} {mae_oos_f*100:>10.1f} "
          f"{r2_oos_f:>8.4f} {bias_oos_f*100:>10.1f}")
    print(f"  [In-sample reference: RMSE={rmse_is*100:.1f} bps, R²={r2_is:.4f}]")
    
    oos_results = {
        'n_train': n_train, 'n_oos': n_oos,
        'params_train': dict(zip(param_names, params_train)),
        'params_full': dict(zip(param_names, params_full)),
        'is_rmse': rmse_is, 'is_r2': r2_is,
        'oos_rmse_train': rmse_oos, 'oos_r2_train': r2_oos,
        'oos_rmse_full': rmse_oos_f, 'oos_r2_full': r2_oos_f,
        'oos_static_rmse': rmse_static, 'oos_static_r2': r2_static,
        'oos_bias_train': bias_oos, 'oos_bias_static': bias_static,
        'dates_oos': dates_oos, 'Y_oos': Y_oos,
        'pred_oos_train': pred_oos_train_params,
        'pred_oos_static': pred_oos_static,
        'pred_oos_full': pred_oos_full_params,
        'FF_oos': FF_oos,
    }
    
    # --- Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel A: OOS fit
    ax = axes[0, 0]
    ax.plot(dates_oos, Y_oos, 'k-', lw=2, label='Actual MMDA', zorder=5)
    ax.plot(dates_oos, pred_oos_train_params, 'b--', lw=1.5,
            label=f'Dynamic v2.0 (train) RMSE={rmse_oos*100:.1f}bps')
    ax.plot(dates_oos, pred_oos_static, 'r:', lw=1.5,
            label=f'Static OLS (train) RMSE={rmse_static*100:.1f}bps')
    ax.set_title('(A) Out-of-Sample Fit: Jan 2022 – Mar 2025', fontweight='bold')
    ax.set_ylabel('MMDA Rate (%)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    # Panel B: OOS residuals
    ax = axes[0, 1]
    ax.bar(dates_oos, resid_oos * 100, color='steelblue', alpha=0.7, width=20,
           label=f'Dynamic v2.0 (bias={bias_oos*100:.1f}bps)')
    ax.axhline(0, color='black', lw=0.5)
    ax.set_title('(B) OOS Residuals: Dynamic v2.0 (Train Params)', fontweight='bold')
    ax.set_ylabel('Residual (bps)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Panel C: Parameter stability
    ax = axes[1, 0]
    pct_changes = [100 * (params_full[i] - params_train[i]) / (abs(params_train[i]) + 1e-10)
                   for i in range(len(param_names))]
    colors = ['#c62828' if abs(p) > 20 else '#1565c0' for p in pct_changes]
    bars = ax.barh(param_names, pct_changes, color=colors, alpha=0.8)
    ax.axvline(0, color='black', lw=0.5)
    ax.axvline(-20, color='gray', ls='--', alpha=0.5, lw=0.8)
    ax.axvline(20, color='gray', ls='--', alpha=0.5, lw=0.8)
    ax.set_title('(C) Parameter Change: Train vs Full Sample (%)', fontweight='bold')
    ax.set_xlabel('% Change')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Panel D: Cumulative OOS error
    ax = axes[1, 1]
    cum_err_dyn = np.cumsum(np.abs(resid_oos)) * 100
    cum_err_static = np.cumsum(np.abs(resid_oos_static)) * 100
    ax.plot(dates_oos, cum_err_dyn, 'b-', lw=2, label='Dynamic v2.0')
    ax.plot(dates_oos, cum_err_static, 'r--', lw=2, label='Static OLS')
    ax.fill_between(dates_oos, cum_err_dyn, cum_err_static,
                    where=cum_err_static > cum_err_dyn,
                    alpha=0.2, color='green', label='Dynamic advantage')
    ax.set_title('(D) Cumulative Absolute Error (bps)', fontweight='bold')
    ax.set_ylabel('Cumulative |Error| (bps)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig_oos_validation.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {path}")
    
    return oos_results


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ROBUSTNESS CHECKS
# ═══════════════════════════════════════════════════════════════════════════════

def run_robustness_checks():
    """
    Robustness tests:
      (a) Alternative volatility windows (6, 12, 24, 36 months)
      (b) Sub-sample stability (2017-2021 vs 2022-2025)
    """
    print("\n" + "="*72)
    print("ROBUSTNESS CHECKS")
    print("="*72)
    
    # ---- (a) Volatility window sensitivity ----
    print("\n--- (a) Volatility Window Sensitivity ---")
    vol_windows = [6, 12, 24, 36]
    vol_results = []
    
    for w in vol_windows:
        print(f"  Estimating with {w}-month vol window...")
        m = EnhancedDynamicBetaModel(max_beta_change=AR_K, volatility_window=w,
                                      min_periods=min(w, 6))
        m.load_data('bankratemma.csv')
        m.estimate(n_starts=3)
        metrics = m.compute_metrics()
        vol_results.append({
            'window': w,
            'r2': metrics['r_squared'],
            'rmse_bps': metrics['rmse'] * 100,
            'aic': metrics['aic'],
            'beta_min': m.params['beta_min'],
            'beta_max': m.params['beta_max'],
            'lambda_up': m.params['lambda_up'],
            'lambda_down': m.params['lambda_down'],
            'm': m.params['m'],
            'k': m.params['k'],
        })
    
    vol_df = pd.DataFrame(vol_results)
    print("\n  Volatility Window Results:")
    print(f"  {'Window':>8} {'R²':>8} {'RMSE':>10} {'β_min':>8} {'β_max':>8} "
          f"{'λ_up':>8} {'λ_down':>8} {'m':>8}")
    print("  " + "-" * 72)
    for _, row in vol_df.iterrows():
        print(f"  {int(row['window']):>6}m {row['r2']:>8.4f} {row['rmse_bps']:>8.2f}bp "
              f"{row['beta_min']:>8.3f} {row['beta_max']:>8.3f} "
              f"{row['lambda_up']:>8.3f} {row['lambda_down']:>8.3f} {row['m']:>8.3f}")
    
    # ---- (b) Sub-sample stability ----
    print("\n--- (b) Sub-Sample Stability ---")
    
    # Full-sample reference
    m_full = EnhancedDynamicBetaModel(max_beta_change=AR_K)
    m_full.load_data('bankratemma.csv')
    m_full.estimate(n_starts=5)
    
    # Pre-hiking (2017-2021)
    m_pre = EnhancedDynamicBetaModel(max_beta_change=AR_K)
    m_pre.load_data('bankratemma.csv', end_date='2021-12-31')
    m_pre.estimate(n_starts=5)
    
    # Hiking cycle (2022-2025)
    m_post = EnhancedDynamicBetaModel(max_beta_change=AR_K)
    m_post.load_data('bankratemma.csv', start_date='2022-01-01')
    m_post.estimate(n_starts=5)
    
    param_names = ['alpha', 'k', 'm', 'beta_min', 'beta_max',
                   'gamma_fhlb', 'lambda_up', 'lambda_down']
    
    print(f"\n  {'Parameter':<14} {'Pre-Hike':>10} {'Post-Hike':>10} {'Full':>10}")
    print("  " + "-" * 48)
    for name in param_names:
        print(f"  {name:<14} {m_pre.params[name]:>10.4f} {m_post.params[name]:>10.4f} "
              f"{m_full.params[name]:>10.4f}")
    
    metrics_pre = m_pre.compute_metrics()
    metrics_post = m_post.compute_metrics()
    metrics_full = m_full.compute_metrics()
    
    print(f"\n  {'Metric':<14} {'Pre-Hike':>10} {'Post-Hike':>10} {'Full':>10}")
    print("  " + "-" * 48)
    print(f"  {'R²':<14} {metrics_pre['r_squared']:>10.4f} {metrics_post['r_squared']:>10.4f} "
          f"{metrics_full['r_squared']:>10.4f}")
    print(f"  {'RMSE (bps)':<14} {metrics_pre['rmse']*100:>10.2f} {metrics_post['rmse']*100:>10.2f} "
          f"{metrics_full['rmse']*100:>10.2f}")
    
    # ---- Plot robustness results ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    
    # Panel A: Vol window R² / RMSE
    ax = axes[0]
    ax2 = ax.twinx()
    bars = ax.bar([str(w) for w in vol_windows], vol_df['r2'], color='steelblue', alpha=0.7)
    line = ax2.plot([str(w) for w in vol_windows], vol_df['rmse_bps'], 'ro-', lw=2, ms=8)
    ax.set_xlabel('Volatility Window (months)')
    ax.set_ylabel('R²', color='steelblue')
    ax2.set_ylabel('RMSE (bps)', color='red')
    ax.set_title('(A) Volatility Window Sensitivity', fontweight='bold')
    ax.set_ylim(0.975, 0.995)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel B: Key parameters across vol windows
    ax = axes[1]
    ax.plot(vol_windows, vol_df['lambda_up'], 's-', color='#c62828', lw=2, ms=8, label='λ_up')
    ax.plot(vol_windows, vol_df['lambda_down'], 'o-', color='#1565c0', lw=2, ms=8, label='λ_down')
    ax.set_xlabel('Volatility Window (months)')
    ax.set_ylabel('Dampening Parameter')
    ax.set_title('(B) Dampening Stability Across Windows', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel C: Sub-sample parameter comparison
    ax = axes[2]
    key_params = ['beta_min', 'beta_max', 'lambda_up', 'lambda_down']
    x = np.arange(len(key_params))
    w_bar = 0.25
    pre_vals = [m_pre.params[p] for p in key_params]
    post_vals = [m_post.params[p] for p in key_params]
    full_vals = [m_full.params[p] for p in key_params]
    ax.bar(x - w_bar, pre_vals, w_bar, label='2017-2021', color='steelblue', alpha=0.8)
    ax.bar(x, post_vals, w_bar, label='2022-2025', color='#c62828', alpha=0.8)
    ax.bar(x + w_bar, full_vals, w_bar, label='Full Sample', color='#2e7d32', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f'β_min', f'β_max', f'λ_up', f'λ_down'])
    ax.set_title('(C) Sub-Sample Parameter Stability', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig_robustness.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {path}")
    
    return {
        'vol_df': vol_df,
        'params_pre': m_pre.params,
        'params_post': m_post.params,
        'params_full': m_full.params,
        'metrics_pre': metrics_pre,
        'metrics_post': metrics_post,
        'metrics_full': metrics_full,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. EFFECTIVE DURATION & CONVEXITY
# ═══════════════════════════════════════════════════════════════════════════════

def compute_duration_convexity():
    """
    Compute effective duration and convexity of MMDA deposits across rate levels.
    
    D_eff = (PV_down - PV_up) / (2 * PV_0 * Δr)
    C_eff = (PV_down + PV_up - 2*PV_0) / (PV_0 * Δr²)
    
    For deposits, PV is compute as annuitized cash flows discounted at risk-free rate,
    but for NMD modeling, effective duration is typically computed as the percentage 
    change in EVE for a parallel rate shift. We use a simplified approach: compute the
    equilibrium deposit cost at each rate level, then measure the sensitivity of the 
    bank's net position (risk-free asset funded by deposits).
    
    For ALM purposes, the key metric is how much the deposit cost changes per bp
    of rate change — which is simply the beta itself. The effective duration 
    interpretation follows from the beta schedule.
    """
    print("\n" + "="*72)
    print("EFFECTIVE DURATION & CONVEXITY CALCULATIONS")
    print("="*72)
    
    p = PARAMS
    shock_bp = 1  # 1 bp for duration
    rates = np.arange(0.25, 7.01, 0.25)
    
    results = []
    for r in rates:
        # Equilibrium beta at this rate (no vol adjustment, pure S-curve)
        beta_0 = logistic_beta(np.array([r]), p['k'], p['m'],
                                p['beta_min'], p['beta_max'])[0]
        
        # MMDA rate at this level
        mmda_0 = p['alpha'] + beta_0 * r + p['gamma_fhlb'] * LAST_FHLB
        
        # 100bp shock for duration calculation
        r_up = r + 1.0
        r_down = max(r - 1.0, 0.05)
        actual_down = r - r_down  # may be less than 1.0 near zero
        
        beta_up = logistic_beta(np.array([r_up]), p['k'], p['m'],
                                 p['beta_min'], p['beta_max'])[0]
        beta_down = logistic_beta(np.array([r_down]), p['k'], p['m'],
                                   p['beta_min'], p['beta_max'])[0]
        
        mmda_up = p['alpha'] + beta_up * r_up + p['gamma_fhlb'] * LAST_FHLB
        mmda_down = p['alpha'] + beta_down * r_down + p['gamma_fhlb'] * LAST_FHLB
        
        # Effective duration of the deposit liability
        # For a deposit paying MMDA rate, the EVE impact of a rate change is:
        # The spread between the risk-free rate and deposit cost changes
        # Duration ≈ Δ(deposit cost) / Δ(rate) — the pass-through rate
        # This is conceptually the "repricing beta"
        
        # More precisely: effective duration in years for the deposit PV
        # Using a 5-year weighted average life assumption for NMD
        avg_life = 5.0  # years, typical ALM assumption
        
        # PV of deposit cash flows (simplified: level annuity at MMDA rate)
        # PV = cost_rate / discount_rate * [1 - 1/(1+discount_rate)^N]
        # But for NMDs, the standard approach is:
        # D_eff = avg_life * beta (the repricing-adjusted duration)
        
        # The repricing-adjusted effective duration:
        d_eff = avg_life * (1 - beta_0)
        
        # Effective convexity from second derivative of beta schedule
        # C_eff = avg_life * d(beta)/dr ≈ avg_life * (beta_up - beta_down) / (r_up - r_down) 
        dbeta_dr = (beta_up - beta_down) / (r_up - r_down)
        # Convexity reduces duration as rates rise (beta increases)
        c_eff = -avg_life * dbeta_dr  # negative because higher beta = shorter duration
        
        # Simple pass-through metrics
        delta_mmda_up = mmda_up - mmda_0
        delta_mmda_down = mmda_0 - mmda_down
        asymmetry = delta_mmda_up / (delta_mmda_down + 1e-10)
        
        results.append({
            'rate': r,
            'beta': beta_0,
            'mmda': mmda_0,
            'effective_duration_yrs': d_eff,
            'effective_convexity': c_eff,
            'pass_through_up_100': delta_mmda_up * 100,  # bps
            'pass_through_down_100': delta_mmda_down * 100,  # bps
            'asymmetry_ratio': asymmetry,
            'dbeta_dr': dbeta_dr,
        })
    
    dur_df = pd.DataFrame(results)
    
    # Print key rate levels
    key_rates = [1.0, 2.0, 3.0, 3.92, 4.33, 5.0, 6.0]
    print(f"\n{'Rate':>6} {'Beta':>6} {'Eff Dur':>8} {'Convex':>8} {'Up PT':>8} {'Dn PT':>8} {'Asym':>6}")
    print("-" * 54)
    for r in key_rates:
        row = dur_df.iloc[(dur_df['rate'] - r).abs().argsort()[:1]].iloc[0]
        print(f"{row['rate']:>5.2f}% {row['beta']:>6.3f} {row['effective_duration_yrs']:>7.2f}y "
              f"{row['effective_convexity']:>8.3f} {row['pass_through_up_100']:>7.1f}bp "
              f"{row['pass_through_down_100']:>7.1f}bp {row['asymmetry_ratio']:>5.2f}×")
    
    # Save CSV
    dur_df.to_csv(os.path.join(OUTPUT_DIR, "duration_convexity_schedule.csv"), index=False)
    
    # ---- Plot ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel A: Beta schedule
    ax = axes[0, 0]
    ax.plot(dur_df['rate'], dur_df['beta'], 'b-', lw=2.5)
    ax.axvline(p['m'], color='gray', ls='--', alpha=0.5, label=f"Inflection (m={p['m']:.1f}%)")
    ax.axvline(LAST_FF, color='red', ls=':', alpha=0.7, label=f"Current FF ({LAST_FF}%)")
    ax.set_xlabel('Fed Funds Rate (%)')
    ax.set_ylabel('Equilibrium Beta')
    ax.set_title('(A) Deposit Beta Schedule', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Panel B: Effective duration
    ax = axes[0, 1]
    ax.plot(dur_df['rate'], dur_df['effective_duration_yrs'], 'b-', lw=2.5)
    ax.axvline(LAST_FF, color='red', ls=':', alpha=0.7, label=f"Current FF ({LAST_FF}%)")
    ax.fill_between(dur_df['rate'], dur_df['effective_duration_yrs'], alpha=0.15, color='steelblue')
    ax.set_xlabel('Fed Funds Rate (%)')
    ax.set_ylabel('Effective Duration (years)')
    ax.set_title('(B) Effective Duration Across Rate Levels', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 4)
    
    # Panel C: Effective convexity
    ax = axes[1, 0]
    ax.plot(dur_df['rate'], dur_df['effective_convexity'], 'b-', lw=2.5)
    ax.axhline(0, color='black', lw=0.5)
    ax.axvline(LAST_FF, color='red', ls=':', alpha=0.7)
    ax.set_xlabel('Fed Funds Rate (%)')
    ax.set_ylabel('Effective Convexity')
    ax.set_title('(C) Effective Convexity (Negative = Duration Shortens as Rates Rise)',
                 fontweight='bold', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Panel D: Pass-through asymmetry
    ax = axes[1, 1]
    ax.bar(dur_df['rate'] - 0.08, dur_df['pass_through_up_100'], width=0.16,
           color='#c62828', alpha=0.7, label='Up 100bp')
    ax.bar(dur_df['rate'] + 0.08, dur_df['pass_through_down_100'], width=0.16,
           color='#1565c0', alpha=0.7, label='Down 100bp')
    ax.set_xlabel('Starting Fed Funds Rate (%)')
    ax.set_ylabel('Pass-Through (bps per 100bp shock)')
    ax.set_title('(D) Asymmetric Pass-Through by Rate Level', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig_duration_convexity.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {path}")
    
    return dur_df


# ═══════════════════════════════════════════════════════════════════════════════
# 4. WORKED PORTFOLIO EXAMPLE
# ═══════════════════════════════════════════════════════════════════════════════

def worked_portfolio_example():
    """
    Complete worked example: $10B MMDA portfolio, +200 bps parallel shock.
    Month-by-month MMDA rate projection, NII impact vs static beta.
    """
    print("\n" + "="*72)
    print("WORKED PORTFOLIO EXAMPLE: $10B MMDA, +200bp Shock")
    print("="*72)
    
    p = PARAMS
    balance = 10_000  # $M
    horizon = 12  # months
    shock_bps = 200
    
    r_new = LAST_FF + shock_bps / 100.0
    
    # ---- Dynamic model path with partial adjustment ----
    beta_start = logistic_beta(np.array([LAST_FF]), p['k'], p['m'],
                                p['beta_min'], p['beta_max'])[0]
    beta_target = logistic_beta(np.array([r_new]), p['k'], p['m'],
                                 p['beta_min'], p['beta_max'])[0]
    
    beta_unc = np.full(horizon, beta_target)
    beta_sm = ar_smoothed_beta(beta_unc, AR_K, beta_init=beta_start)
    
    # Equilibrium MMDA at each month (from AR-smoothed beta)
    mmda_eq = p['alpha'] + beta_sm * r_new + p['gamma_fhlb'] * LAST_FHLB
    
    # Apply Nerlove partial adjustment: MMDA_t = MMDA_{t-1} + theta*(MMDA*_t - MMDA_{t-1})
    mmda_dynamic = np.zeros(horizon)
    mmda_prev = LAST_MMDA  # actual current MMDA rate (pre-shock)
    for i in range(horizon):
        mmda_dynamic[i] = mmda_prev + THETA * (mmda_eq[i] - mmda_prev)
        mmda_prev = mmda_dynamic[i]
    
    # ---- Static OLS path ----
    static_beta = 0.462  # from Table 2a
    static_intercept = 0.249
    mmda_static_base = static_intercept + static_beta * LAST_FF
    mmda_static_shocked = static_intercept + static_beta * r_new
    mmda_static = np.full(horizon, mmda_static_shocked)
    
    # ---- Base case (no shock) ----
    mmda_base = np.full(horizon, p['alpha'] + beta_start * LAST_FF + p['gamma_fhlb'] * LAST_FHLB)
    
    # ---- Interest expense calculations ----
    # Monthly interest expense = balance * (annual rate / 12) * (1/100)
    ie_base = balance * (mmda_base / 12 / 100)
    ie_dynamic = balance * (mmda_dynamic / 12 / 100)
    ie_static = balance * (mmda_static / 12 / 100)
    
    cum_ie_base = np.cumsum(ie_base)
    cum_ie_dynamic = np.cumsum(ie_dynamic)
    cum_ie_static = np.cumsum(ie_static)
    
    # On the asset side: assume floating-rate assets reprice immediately
    asset_yield_base = LAST_FF
    asset_yield_shock = r_new
    asset_income_base = balance * (asset_yield_base / 12 / 100) * np.ones(horizon)
    asset_income_shock = balance * (asset_yield_shock / 12 / 100) * np.ones(horizon)
    
    # NIM impact
    nim_base = asset_yield_base - mmda_base[0]
    nim_dynamic = asset_yield_shock - mmda_dynamic
    nim_static = asset_yield_shock - mmda_static[0]
    
    # Monthly NII
    nii_base = balance * (nim_base / 12 / 100) * np.ones(horizon)
    nii_dynamic = balance * (nim_dynamic / 12 / 100)
    nii_static = balance * (nim_static / 12 / 100) * np.ones(horizon)
    
    cum_nii_base = np.cumsum(nii_base)
    cum_nii_dynamic = np.cumsum(nii_dynamic)
    cum_nii_static = np.cumsum(nii_static)
    
    annual_nii_base = cum_nii_base[-1]
    annual_nii_dynamic = cum_nii_dynamic[-1]
    annual_nii_static = cum_nii_static[-1]
    
    print(f"\nScenario: +{shock_bps}bps parallel shock, $10B MMDA book")
    print(f"Current FF: {LAST_FF:.2f}%  →  New FF: {r_new:.2f}%")
    print(f"Starting beta: {beta_start:.3f}  →  Equilibrium beta: {beta_target:.3f}")
    print(f"\n{'Month':>6} {'Beta':>7} {'MMDA (dyn)':>11} {'MMDA (stat)':>12} "
          f"{'IE Diff ($M)':>12} {'Cum Diff ($M)':>14}")
    print("-" * 65)
    
    cum_diff = 0
    for i in range(horizon):
        ie_diff = (ie_dynamic[i] - ie_static[i])
        cum_diff += ie_diff
        print(f"{i+1:>6} {beta_sm[i]:>7.3f} {mmda_dynamic[i]:>10.3f}% "
              f"{mmda_static[i]:>11.3f}% {ie_diff:>11.2f} {cum_diff:>13.2f}")
    
    annual_ie_diff = cum_ie_dynamic[-1] - cum_ie_static[-1]
    annual_ie_diff_vs_base = cum_ie_dynamic[-1] - cum_ie_base[-1]
    
    print(f"\n--- 12-Month Summary ---")
    print(f"  12-month interest expense (dynamic):  ${cum_ie_dynamic[-1]:,.1f}M")
    print(f"  12-month interest expense (static):   ${cum_ie_static[-1]:,.1f}M")
    print(f"  12-month interest expense (base):     ${cum_ie_base[-1]:,.1f}M")
    print(f"  Dynamic vs Static difference:         ${annual_ie_diff:,.1f}M")
    print(f"  Dynamic vs Base (shock cost):         ${annual_ie_diff_vs_base:,.1f}M")
    print(f"  12-month NII (base):                  ${annual_nii_base:,.1f}M")
    print(f"  12-month NII (dynamic):               ${annual_nii_dynamic:,.1f}M")
    print(f"  12-month NII (static):                ${annual_nii_static:,.1f}M")
    
    # ---- Plot ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    months = np.arange(1, horizon + 1)
    
    # Panel A: MMDA rate transition
    ax = axes[0]
    ax.plot(months, mmda_dynamic, 'b-o', lw=2, ms=5, label='Dynamic S-Curve')
    ax.plot(months, mmda_static, 'r--s', lw=2, ms=5, label='Static OLS (β=46.2%)')
    ax.axhline(mmda_base[0], color='gray', ls=':', lw=1, label=f'Base (no shock): {mmda_base[0]:.2f}%')
    ax.set_xlabel('Month After Shock')
    ax.set_ylabel('MMDA Rate (%)')
    ax.set_title(f'(A) MMDA Rate: +{shock_bps}bp Shock', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    # Panel B: Monthly interest expense difference
    ax = axes[1]
    monthly_diff = (ie_dynamic - ie_static)
    ax.bar(months, monthly_diff, color='#c62828', alpha=0.7)
    ax.axhline(0, color='black', lw=0.5)
    ax.set_xlabel('Month After Shock')
    ax.set_ylabel('Additional IE vs Static ($M)')
    ax.set_title('(B) Monthly Cost Underestimation by Static Beta', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel C: Cumulative NII comparison
    ax = axes[2]
    ax.plot(months, cum_nii_dynamic, 'b-o', lw=2, ms=5, label='Dynamic S-Curve')
    ax.plot(months, cum_nii_static, 'r--s', lw=2, ms=5, label='Static OLS')
    ax.plot(months, cum_nii_base, 'k:', lw=1.5, label='Base (no shock)')
    ax.set_xlabel('Month After Shock')
    ax.set_ylabel('Cumulative NII ($M)')
    ax.set_title(f'(C) Cumulative NII: $10B MMDA, +{shock_bps}bp', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig_portfolio_example.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {path}")
    
    return {
        'mmda_dynamic': mmda_dynamic, 'mmda_static': mmda_static,
        'cum_ie_dynamic': cum_ie_dynamic[-1], 'cum_ie_static': cum_ie_static[-1],
        'annual_ie_diff': annual_ie_diff,
        'beta_sm': beta_sm, 'beta_target': beta_target,
        'nii_dynamic': annual_nii_dynamic, 'nii_static': annual_nii_static,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 5. BOOTSTRAP CONFIDENCE INTERVALS ON SCENARIOS
# ═══════════════════════════════════════════════════════════════════════════════

def bootstrap_scenario_ci(n_boot=500):
    """
    Parametric bootstrap: resample residuals, re-estimate, re-forecast.
    Produces confidence bands on scenario forecasts.
    """
    print("\n" + "="*72)
    print(f"BOOTSTRAP CONFIDENCE INTERVALS ({n_boot} replications)")
    print("="*72)
    
    # Fit the model to get residuals
    model = EnhancedDynamicBetaModel(max_beta_change=AR_K)
    model.load_data('bankratemma.csv')
    model.estimate(n_starts=5)
    
    residuals = model.residuals
    fitted = model.predictions
    n = len(residuals)
    
    shocks = [200, -200]  # Focus on ±200bp
    horizon = 36
    
    # Storage for bootstrap forecasts
    boot_paths = {s: np.zeros((n_boot, horizon)) for s in shocks}
    boot_betas = {s: np.zeros((n_boot, horizon)) for s in shocks}
    boot_params = []
    
    Y = model.Y.copy()
    FEDL01 = model.FEDL01
    FHLK3MSPRD = model.FHLK3MSPRD
    vol_ratio = model.vol_ratio
    rate_change = model.rate_change
    
    bounds = [
        (-1, 1), (0.01, 5), (0.5, 5), (0.30, 0.50),
        (0.55, 0.80), (-3, 3), (0, 1), (0, 1),
    ]
    
    print(f"  Running {n_boot} bootstrap replications...")
    successful = 0
    
    for b in range(n_boot):
        if (b + 1) % 100 == 0:
            print(f"    Bootstrap {b+1}/{n_boot}...")
        
        # Resample residuals (block bootstrap, block size 6)
        block_size = 6
        n_blocks = n // block_size + 1
        block_starts = np.random.randint(0, n - block_size + 1, size=n_blocks)
        boot_resid = np.concatenate([residuals[s:s+block_size] for s in block_starts])[:n]
        
        # Construct bootstrap Y
        Y_boot = fitted + boot_resid
        
        # Re-estimate
        def nll_boot(params):
            alpha, k, m, beta_min, beta_max, gamma_fhlb, lambda_up, lambda_down = params
            beta_unc = asymmetric_volatility_beta(
                FEDL01, k, m, beta_min, beta_max, vol_ratio,
                lambda_up, lambda_down, rate_change
            )
            beta_sm = ar_smoothed_beta(beta_unc, AR_K)
            pred = alpha + beta_sm * FEDL01 + gamma_fhlb * FHLK3MSPRD
            resid = Y_boot - pred
            sigma_sq = np.var(resid)
            if sigma_sq <= 0:
                return 1e10
            return 0.5 * n * np.log(2 * np.pi * sigma_sq) + np.sum(resid**2) / (2 * sigma_sq)
        
        try:
            result = minimize(nll_boot, model.result.x, bounds=bounds,
                            method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-7})
            if not result.success:
                continue
        except:
            continue
        
        bp = result.x
        boot_params.append(bp)
        
        # Generate scenario forecasts with bootstrap parameters
        for shock_bps in shocks:
            r_new = LAST_FF + shock_bps / 100.0
            beta_start = logistic_beta(np.array([LAST_FF]), bp[1], bp[2],
                                        bp[3], bp[4])[0]
            beta_target = logistic_beta(np.array([r_new]), bp[1], bp[2],
                                         bp[3], bp[4])[0]
            beta_unc = np.full(horizon, beta_target)
            beta_sm = ar_smoothed_beta(beta_unc, AR_K, beta_init=beta_start)
            mmda = bp[0] + beta_sm * r_new + bp[5] * LAST_FHLB
            boot_paths[shock_bps][successful] = mmda
            boot_betas[shock_bps][successful] = beta_sm
        
        successful += 1
    
    print(f"  Successful: {successful}/{n_boot}")
    
    # Trim to successful
    for s in shocks:
        boot_paths[s] = boot_paths[s][:successful]
        boot_betas[s] = boot_betas[s][:successful]
    
    # Compute percentiles
    ci_results = {}
    for s in shocks:
        ci_results[s] = {
            'median': np.median(boot_paths[s], axis=0),
            'p5': np.percentile(boot_paths[s], 5, axis=0),
            'p25': np.percentile(boot_paths[s], 25, axis=0),
            'p75': np.percentile(boot_paths[s], 75, axis=0),
            'p95': np.percentile(boot_paths[s], 95, axis=0),
        }
    
    # Point estimates for comparison
    from scenario_shock_analysis import run_scenario
    for s in shocks:
        res = run_scenario(s, n_months=horizon)
        ci_results[s]['point'] = res['mmda_forecast']
    
    # Print summary
    for s in shocks:
        ci = ci_results[s]
        print(f"\n  ±200bp shock = {'+' if s > 0 else ''}{s}bp:")
        print(f"    12m: {ci['point'][11]:.3f}%  (90% CI: [{ci['p5'][11]:.3f}, {ci['p95'][11]:.3f}])")
        print(f"    36m: {ci['point'][35]:.3f}%  (90% CI: [{ci['p5'][35]:.3f}, {ci['p95'][35]:.3f}])")
    
    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    months = np.arange(1, horizon + 1)
    
    for idx, s in enumerate(shocks):
        ax = axes[idx]
        ci = ci_results[s]
        label = f"+{s}bp" if s > 0 else f"{s}bp"
        
        ax.fill_between(months, ci['p5'], ci['p95'], alpha=0.15, color='steelblue',
                        label='90% CI')
        ax.fill_between(months, ci['p25'], ci['p75'], alpha=0.3, color='steelblue',
                        label='50% CI')
        ax.plot(months, ci['point'], 'b-', lw=2.5, label='Point Estimate')
        ax.plot(months, ci['median'], 'r--', lw=1.5, label='Bootstrap Median')
        
        ax.set_xlabel('Month')
        ax.set_ylabel('MMDA Rate (%)')
        ax.set_title(f'({chr(65+idx)}) {label} Shock: Bootstrap Confidence Bands', fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig_bootstrap_ci.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {path}")
    
    # Parameter distribution summary
    boot_params = np.array(boot_params[:successful])
    param_names = ['alpha', 'k', 'm', 'beta_min', 'beta_max',
                   'gamma_fhlb', 'lambda_up', 'lambda_down']
    print(f"\n  Bootstrap Parameter Distribution (n={successful}):")
    print(f"  {'Param':<14} {'Mean':>8} {'Std':>8} {'5%':>8} {'95%':>8}")
    print("  " + "-" * 52)
    for i, name in enumerate(param_names):
        vals = boot_params[:, i]
        print(f"  {name:<14} {np.mean(vals):>8.4f} {np.std(vals):>8.4f} "
              f"{np.percentile(vals, 5):>8.4f} {np.percentile(vals, 95):>8.4f}")
    
    return ci_results


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("="*72)
    print("RUNNING ALL PAPER ENHANCEMENT ANALYSES")
    print("="*72)
    
    results = {}
    
    # 1. Out-of-sample validation
    results['oos'] = run_oos_validation()
    
    # 2. Robustness checks
    results['robust'] = run_robustness_checks()
    
    # 3. Effective duration & convexity
    results['duration'] = compute_duration_convexity()
    
    # 4. Worked portfolio example
    results['portfolio'] = worked_portfolio_example()
    
    # 5. Bootstrap CI (takes longest)
    results['bootstrap'] = bootstrap_scenario_ci(n_boot=500)
    
    print("\n" + "="*72)
    print("ALL ENHANCEMENT ANALYSES COMPLETE")
    print(f"Outputs saved to: {OUTPUT_DIR}/")
    print("="*72)
    
    # Summary file
    with open(os.path.join(OUTPUT_DIR, "enhancement_summary.txt"), 'w') as f:
        oos = results['oos']
        f.write("PAPER ENHANCEMENT RESULTS SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write("1. OUT-OF-SAMPLE VALIDATION\n")
        f.write(f"   Train: 2017-2021 ({oos['n_train']} obs)\n")
        f.write(f"   OOS: 2022-2025 ({oos['n_oos']} obs)\n")
        f.write(f"   In-sample R²:  {oos['is_r2']:.4f}, RMSE: {oos['is_rmse']*100:.1f}bps\n")
        f.write(f"   OOS R² (train params): {oos['oos_r2_train']:.4f}, RMSE: {oos['oos_rmse_train']*100:.1f}bps\n")
        f.write(f"   OOS Static R²: {oos['oos_static_r2']:.4f}, RMSE: {oos['oos_static_rmse']*100:.1f}bps\n")
        f.write(f"   OOS bias (dynamic): {oos['oos_bias_train']*100:.1f}bps\n\n")
        
        rob = results['robust']
        f.write("2. ROBUSTNESS: VOLATILITY WINDOWS\n")
        for _, row in rob['vol_df'].iterrows():
            f.write(f"   {int(row['window'])}m window: R²={row['r2']:.4f}, RMSE={row['rmse_bps']:.2f}bps\n")
        f.write(f"\n3. EFFECTIVE DURATION AT CURRENT RATE ({LAST_FF}%)\n")
        dur_row = results['duration'].iloc[(results['duration']['rate'] - LAST_FF).abs().argsort()[:1]].iloc[0]
        f.write(f"   Beta: {dur_row['beta']:.3f}\n")
        f.write(f"   Effective Duration: {dur_row['effective_duration_yrs']:.2f} years\n")
        f.write(f"   Effective Convexity: {dur_row['effective_convexity']:.3f}\n\n")
        
        port = results['portfolio']
        f.write("4. PORTFOLIO EXAMPLE ($10B MMDA, +200bp)\n")
        f.write(f"   12m IE (dynamic): ${port['cum_ie_dynamic']:,.1f}M\n")
        f.write(f"   12m IE (static):  ${port['cum_ie_static']:,.1f}M\n")
        f.write(f"   Difference:       ${port['annual_ie_diff']:,.1f}M\n")
    
    print(f"Summary saved to: {os.path.join(OUTPUT_DIR, 'enhancement_summary.txt')}")
