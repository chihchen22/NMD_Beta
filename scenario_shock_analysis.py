"""
Scenario Analysis: Parallel Rate Shocks (+/- 100, 200, 300 bps)
================================================================

Generates forecasted MMDA rate paths under instantaneous parallel shocks,
highlighting the asymmetry between upward and downward rate movements.

Produces:
  outputs/scenario_analysis/
    fig_scenario_shock_paths.png       — Rate path forecasts for each shock
    fig_scenario_asymmetry.png         — Asymmetry analysis (up vs down)
    fig_scenario_beta_convergence.png  — Beta transition dynamics
    scenario_shock_data.csv            — Full numerical output

Author: Chih L. Chen
Date: February 2026
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FormatStrFormatter

# ── Add project root to path ────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from enhanced_dynamic_beta_model import (
    logistic_beta, asymmetric_volatility_beta, ar_smoothed_beta
)

# ── Configuration ────────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join("outputs", "scenario_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FORECAST_MONTHS = 36
SHOCKS_BPS = [-300, -200, -100, 0, 100, 200, 300]

# Model parameters from v2.0 estimation
PARAMS = {
    'alpha': 0.073,
    'k': 0.566,
    'm': 3.919,
    'beta_min': 0.433,
    'beta_max': 0.800,
    'gamma_fhlb': 1.049,
    'lambda_up': 0.255,
    'lambda_down': 0.223,
}
AR_K = 0.02  # AR smoothing max delta per period

# Latest observed data (March 2025)
LAST_FF = 4.33
LAST_MMDA = 2.495
LAST_FHLB = 0.1834

# Trailing 24 months of Fed Funds rate changes (Apr 2023 - Mar 2025)
# Used to compute rolling volatility in forward projections
TRAILING_DELTA_R = [
    0.1770, 0.2159, 0.0298, 0.0413, 0.2130, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    -0.2000, -0.3000, -0.1842, -0.1610, -0.1548, 0.0, 0.0,
]

# Current volatility state (March 2025)
LAST_VOL_24M = 0.1208
LAST_VOL_STAR = 0.1232
N_VOL_OBS = 130  # number of historical vol_24m observations for expanding mean

# Max historical 1-month Fed Funds move: 95.8bp (March 2020, COVID emergency cut).
# For vol computation, the shock month's Δr is capped at this value so that
# hypothetical >100bp instant shocks don't produce out-of-sample vol_ratios.
# All shocks ≥96bp get the same dampening in the shock month.
MAX_HIST_MONTHLY_MOVE = 0.9582

# Partial adjustment speed (Nerlove, 1958).
# Estimated from historical data via OLS: ΔMMDA_t = θ(MMDA*_t - MMDA_{t-1}).
# θ = 0.47 (SE = 0.049, t = 9.71). About 47% of the gap between the current
# MMDA rate and the model-implied equilibrium closes each month.
THETA = 0.4707

# Style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'figure.facecolor': 'white',
})

SHOCK_COLORS = {
    -300: '#1a237e', -200: '#1565c0', -100: '#42a5f5',
    0: '#616161',
    100: '#ef5350', 200: '#c62828', 300: '#7f0000',
}


# ═══════════════════════════════════════════════════════════════════════════════
# SCENARIO ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def compute_equilibrium_beta(rate, vol_ratio=1.0, rate_change_sign=0):
    """Compute the equilibrium (unconstrained) beta at a given rate level.
    
    For scenario projections, vol dampening is applied only in the shock
    month (when rate_change_sign != 0). In subsequent flat months,
    vol_adj = 1.0 so beta converges to the pure S-curve level.
    """
    p = PARAMS
    base = logistic_beta(np.array([rate]), p['k'], p['m'], p['beta_min'], p['beta_max'])[0]
    if rate_change_sign > 0:
        lam = p['lambda_up']
    elif rate_change_sign < 0:
        lam = p['lambda_down']
    else:
        return base  # flat month: no vol dampening
    vol_adj = np.clip(1 - lam * vol_ratio, 0.5, 1.5)
    return base * vol_adj


def run_scenario(shock_bps, n_months=FORECAST_MONTHS):
    """
    Run a parallel shock scenario with shock-month volatility dampening.
    
    Month 0: FF rate jumps by shock_bps instantaneously.
    Months 1-n: Rate stays flat at the new level.
    
    Vol dampening logic:
      - Shock month (month 1): vol_ratio is computed by inserting the shock
        into the trailing 24-month window. The shock's Δr contribution is
        capped at MAX_HIST_MONTHLY_MOVE (96bp) so that hypothetical >100bp
        shocks don't produce out-of-sample vol inflation. The appropriate
        lambda (up or down) is applied.
      - Flat months (months 2-n): vol_adj = 1.0. No dampening. Beta
        converges to the pure S-curve equilibrium level.
    
    Returns dict with monthly series for rates, betas, MMDA forecasts.
    """
    p = PARAMS
    r_new = LAST_FF + shock_bps / 100.0
    r_new = max(r_new, 0.05)  # floor at 5 bps
    
    # Starting beta: equilibrium at current rate (pre-shock)
    beta_start = logistic_beta(np.array([LAST_FF]), p['k'], p['m'],
                                p['beta_min'], p['beta_max'])[0]
    
    # S-curve equilibrium beta at new rate (no vol adjustment)
    beta_level = logistic_beta(np.array([r_new]), p['k'], p['m'],
                                p['beta_min'], p['beta_max'])[0]
    
    # Compute shock-month vol_ratio
    shock_delta = shock_bps / 100.0
    if shock_delta != 0:
        # Cap the shock's vol contribution at max historical 1-month move
        capped_delta = np.sign(shock_delta) * min(abs(shock_delta), MAX_HIST_MONTHLY_MOVE)
        window = list(TRAILING_DELTA_R)
        window.append(capped_delta)
        window.pop(0)
        vol_24m_shock = np.std(window, ddof=1)
        vol_star_shock = (LAST_VOL_STAR * N_VOL_OBS + vol_24m_shock) / (N_VOL_OBS + 1)
        vol_ratio_shock = vol_24m_shock / vol_star_shock
    else:
        vol_ratio_shock = LAST_VOL_24M / LAST_VOL_STAR  # ~0.98
    
    # Build unconstrained beta path:
    #   Month 0 (shock month): apply vol dampening with lambda_up or lambda_down
    #   Months 1+: pure S-curve level (vol_adj = 1.0)
    beta_unconstrained = np.full(n_months, beta_level)  # flat at S-curve target
    
    vol_ratios = np.full(n_months, 1.0)  # store for diagnostics
    
    if shock_delta != 0:
        # Shock month gets vol dampening
        lam = p['lambda_up'] if shock_delta > 0 else p['lambda_down']
        vol_adj = np.clip(1 - lam * vol_ratio_shock, 0.5, 1.5)
        beta_unconstrained[0] = beta_level * vol_adj
        vol_ratios[0] = vol_ratio_shock
    
    # Apply AR smoothing
    beta_smoothed = ar_smoothed_beta(beta_unconstrained, AR_K, beta_init=beta_start)
    
    # Compute MMDA equilibrium path (what the model says rate should be)
    mmda_star = p['alpha'] + beta_smoothed * r_new + p['gamma_fhlb'] * LAST_FHLB
    
    # Apply partial adjustment: MMDA_t = MMDA_{t-1} + θ(MMDA*_t - MMDA_{t-1})
    mmda_forecast = np.zeros(n_months)
    mmda_prev = LAST_MMDA
    for t in range(n_months):
        mmda_forecast[t] = mmda_prev + THETA * (mmda_star[t] - mmda_prev)
        mmda_prev = mmda_forecast[t]
    
    # Compute "equilibrium" MMDA (where it settles: pure S-curve, no vol adj)
    mmda_eq = p['alpha'] + beta_level * r_new + p['gamma_fhlb'] * LAST_FHLB
    
    return {
        'shock_bps': shock_bps,
        'r_new': r_new,
        'beta_start': beta_start,
        'beta_target': beta_level,  # S-curve level (no vol adj)
        'beta_unconstrained': beta_unconstrained,
        'beta_smoothed': beta_smoothed,
        'mmda_forecast': mmda_forecast,
        'mmda_equilibrium': mmda_eq,
        'mmda_start': LAST_MMDA,
        'vol_ratios': vol_ratios,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# RUN ALL SCENARIOS
# ═══════════════════════════════════════════════════════════════════════════════

def run_all_scenarios():
    """Run shocks and return dict of results."""
    results = {}
    for bps in SHOCKS_BPS:
        results[bps] = run_scenario(bps)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Scenario Rate Path Forecasts
# ═══════════════════════════════════════════════════════════════════════════════

def fig_shock_paths(results, path):
    """2-panel: (A) MMDA rate forecasts, (B) cumulative deposit cost change."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    
    fcast_dates = pd.date_range(start="2025-04-30", periods=FORECAST_MONTHS, freq="ME")
    hist_dates = pd.date_range(start="2024-04-30", periods=12, freq="ME")
    
    # Load historical for context
    try:
        raw = pd.read_csv("bankratemma.csv")
        raw['EOM_Dt'] = pd.to_datetime(raw['EOM_Dt'])
        raw = raw.set_index('EOM_Dt').sort_index()
        hist_mmda = raw['ILMDHYLD'].iloc[-12:].values
        hist_dates = raw.index[-12:]
    except Exception:
        hist_mmda = np.full(12, LAST_MMDA)
    
    # Panel A: MMDA rate paths
    ax = axes[0]
    ax.plot(hist_dates, hist_mmda, 'k-', lw=2, label='Historical', zorder=5)
    ax.axvline(pd.Timestamp("2025-03-31"), color='gray', ls='--', alpha=0.5, lw=0.8)
    
    for bps in SHOCKS_BPS:
        res = results[bps]
        label = f"+{bps}" if bps > 0 else f"{bps}" if bps < 0 else "Base (flat)"
        label += " bps"
        lw = 2.2 if bps == 0 else 1.5
        ls = '--' if bps == 0 else '-'
        ax.plot(fcast_dates, res['mmda_forecast'], color=SHOCK_COLORS[bps],
                ls=ls, lw=lw, label=label, alpha=0.9)
    
    ax.set_xlabel("Date")
    ax.set_ylabel("MMDA Rate (%)")
    ax.set_title("(A) Forecasted MMDA Rates Under Parallel Shocks", fontweight='bold')
    ax.legend(loc='best', fontsize=8, ncol=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
    
    # Panel B: Cumulative change from base (deposit cost impact)
    ax = axes[1]
    base_path = results[0]['mmda_forecast']
    
    for bps in SHOCKS_BPS:
        if bps == 0:
            continue
        res = results[bps]
        delta_from_base = res['mmda_forecast'] - base_path
        label = f"+{bps}" if bps > 0 else f"{bps}"
        label += " bps"
        ax.plot(fcast_dates, delta_from_base * 100, color=SHOCK_COLORS[bps],
                lw=1.5, label=label, alpha=0.9)
    
    ax.axhline(0, color='gray', ls='-', lw=0.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Change vs. Base (bps)")
    ax.set_title("(B) Incremental Deposit Cost vs. Flat Rates", fontweight='bold')
    ax.legend(loc='best', fontsize=8, ncol=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
    
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Asymmetry Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def fig_asymmetry(results, path):
    """
    4-panel analysis of up-vs-down shock asymmetry.
    (A) Mirror comparison: +200 vs -200
    (B) MMDA change per 100 bps at selected horizons
    (C) Beta convergence: up vs down 200 bps
    (D) Pass-through ratio time profile
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fcast_dates = pd.date_range(start="2025-04-30", periods=FORECAST_MONTHS, freq="ME")
    months = np.arange(1, FORECAST_MONTHS + 1)
    
    # ── Panel A: Mirror comparison (+200 vs -200) ──
    ax = axes[0, 0]
    base = results[0]
    up200 = results[200]
    dn200 = results[-200]
    
    # Plot MMDA paths
    ax.plot(months, up200['mmda_forecast'], color=SHOCK_COLORS[200], lw=2,
            label='+200 bps (FF={:.2f}%)'.format(up200['r_new']))
    ax.plot(months, dn200['mmda_forecast'], color=SHOCK_COLORS[-200], lw=2,
            label='-200 bps (FF={:.2f}%)'.format(dn200['r_new']))
    ax.plot(months, base['mmda_forecast'], color=SHOCK_COLORS[0], lw=1.5,
            ls='--', label=f"Base (FF={LAST_FF:.2f}%)")
    
    # Equilibrium lines
    ax.axhline(up200['mmda_equilibrium'], color=SHOCK_COLORS[200], ls=':', alpha=0.4)
    ax.axhline(dn200['mmda_equilibrium'], color=SHOCK_COLORS[-200], ls=':', alpha=0.4)
    
    ax.set_xlabel("Months After Shock")
    ax.set_ylabel("MMDA Rate (%)")
    ax.set_title("(A) Symmetric Shock, Asymmetric Response", fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # ── Panel B: Pass-through at key horizons ──
    ax = axes[0, 1]
    horizons = [1, 3, 6, 12, 24, 36]
    shock_pairs = [(-300, 300), (-200, 200), (-100, 100)]
    
    bar_width = 0.25
    x = np.arange(len(horizons))
    
    for i, (dn_bps, up_bps) in enumerate(shock_pairs):
        up_res = results[up_bps]
        dn_res = results[dn_bps]
        
        # Pass-through = (MMDA change from base) / (FF change)
        up_pt = []
        dn_pt = []
        for h in horizons:
            idx = h - 1
            up_mmda_chg = up_res['mmda_forecast'][idx] - base['mmda_forecast'][idx]
            dn_mmda_chg = dn_res['mmda_forecast'][idx] - base['mmda_forecast'][idx]
            ff_chg_up = up_bps / 100.0
            ff_chg_dn = dn_bps / 100.0
            up_pt.append(up_mmda_chg / ff_chg_up * 100)
            dn_pt.append(abs(dn_mmda_chg / ff_chg_dn) * 100)
        
        offset = (i - 1) * bar_width
        ax.bar(x + offset - bar_width/4, up_pt, bar_width/2,
               color=SHOCK_COLORS[up_bps], alpha=0.8,
               label=f'+{up_bps} (up)' if i == 0 else f'+{up_bps}')
        ax.bar(x + offset + bar_width/4, dn_pt, bar_width/2,
               color=SHOCK_COLORS[dn_bps], alpha=0.8,
               label=f'{dn_bps} (down)' if i == 0 else f'{dn_bps}')
    
    ax.set_xticks(x)
    ax.set_xticklabels([f'{h}m' for h in horizons])
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Pass-Through (%)")
    ax.set_title("(B) Cumulative Pass-Through by Horizon", fontweight='bold')
    ax.legend(fontsize=7, ncol=3)
    ax.grid(True, alpha=0.3, axis='y')
    
    # ── Panel C: Beta convergence paths ──
    ax = axes[1, 0]
    for bps in [300, 200, 100, -100, -200, -300]:
        res = results[bps]
        label = f"+{bps}" if bps > 0 else f"{bps}"
        ax.plot(months, res['beta_smoothed'], color=SHOCK_COLORS[bps],
                lw=1.5, label=f'{label} bps')
    
    ax.plot(months, base['beta_smoothed'], color=SHOCK_COLORS[0], lw=1.5,
            ls='--', label='Base')
    
    ax.set_xlabel("Months After Shock")
    ax.set_ylabel("Effective Beta")
    ax.set_title("(C) Beta Convergence Under Shocks", fontweight='bold')
    ax.legend(fontsize=8, ncol=2)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.grid(True, alpha=0.3)
    
    # ── Panel D: Asymmetry ratio ──
    ax = axes[1, 1]
    for mag in [100, 200, 300]:
        up_res = results[mag]
        dn_res = results[-mag]
        
        up_delta = up_res['mmda_forecast'] - base['mmda_forecast']
        dn_delta = base['mmda_forecast'] - dn_res['mmda_forecast']
        
        # Asymmetry ratio: how much bigger is the up-change vs down-change
        # >1 means upward shocks pass through more; <1 means downward pass through more
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(dn_delta > 0.0001, up_delta / dn_delta, np.nan)
        
        ax.plot(months, ratio, color=SHOCK_COLORS[mag], lw=1.5,
                label=f'±{mag} bps')
    
    ax.axhline(1.0, color='gray', ls='--', lw=1, alpha=0.5)
    ax.set_xlabel("Months After Shock")
    ax.set_ylabel("Up/Down Pass-Through Ratio")
    ax.set_title("(D) Asymmetry Ratio (>1 = Up Passes Through More)", fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 2.0)
    
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Beta Convergence Detail
# ═══════════════════════════════════════════════════════════════════════════════

def fig_beta_convergence(results, path):
    """
    2-panel: (A) Smoothed vs unconstrained beta for ±200 bps
             (B) Months to reach 90%/95%/99% of equilibrium
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    months = np.arange(1, FORECAST_MONTHS + 1)
    
    # Panel A: Smoothed vs unconstrained for ±200
    ax = axes[0]
    for bps in [200, -200]:
        res = results[bps]
        clr = SHOCK_COLORS[bps]
        label_dir = "Up +200" if bps > 0 else "Down -200"
        ax.plot(months, res['beta_smoothed'], color=clr, lw=2,
                label=f'{label_dir} (smoothed)')
        ax.plot(months, res['beta_unconstrained'], color=clr, lw=1, ls=':',
                alpha=0.6, label=f'{label_dir} (unconstrained)')
        ax.axhline(res['beta_target'], color=clr, ls='--', alpha=0.3, lw=0.8)
    
    base = results[0]
    ax.axhline(base['beta_target'], color='gray', ls='--', alpha=0.4, lw=0.8,
               label=f'Base equil. ({base["beta_target"]:.3f})')
    
    ax.set_xlabel("Months After Shock")
    ax.set_ylabel("Effective Beta")
    ax.set_title("(A) AR-Smoothed vs. Unconstrained Beta (±200 bps)", fontweight='bold')
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.grid(True, alpha=0.3)
    
    # Panel B: Time to convergence table
    ax = axes[1]
    ax.axis('off')
    
    thresholds = [0.90, 0.95, 0.99]
    rows = [["Shock", "Target β", "90%", "95%", "99%"]]
    
    for bps in [-300, -200, -100, 100, 200, 300]:
        res = results[bps]
        beta_start_val = res['beta_start']
        beta_target_val = res['beta_target']
        total_delta = beta_target_val - beta_start_val
        
        row = [f"{'+' if bps > 0 else ''}{bps} bps", f"{beta_target_val:.3f}"]
        for thr in thresholds:
            if abs(total_delta) < 0.001:
                row.append("< 1")
                continue
            pct_achieved = np.abs(res['beta_smoothed'] - beta_start_val) / abs(total_delta)
            indices = np.where(pct_achieved >= thr)[0]
            if len(indices) > 0:
                row.append(f"{indices[0] + 1}")
            else:
                row.append(f"> {FORECAST_MONTHS}")
        rows.append(row)
    
    table = ax.table(cellText=rows[1:], colLabels=rows[0],
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)
    
    # Style header
    for j in range(len(rows[0])):
        table[0, j].set_facecolor('#2c3e6b')
        table[0, j].set_text_props(color='white', fontweight='bold')
    
    # Alternate row colors
    for i in range(1, len(rows)):
        color = '#f7f8fb' if i % 2 == 0 else 'white'
        for j in range(len(rows[0])):
            table[i, j].set_facecolor(color)
    
    ax.set_title("(B) Months to Convergence (% of Total β Adjustment)",
                 fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════════

def generate_summary_table(results):
    """Generate a summary CSV with key metrics for each scenario."""
    base = results[0]
    rows = []
    for bps in SHOCKS_BPS:
        res = results[bps]
        mmda_1m = res['mmda_forecast'][0]
        mmda_6m = res['mmda_forecast'][5]
        mmda_12m = res['mmda_forecast'][11]
        mmda_36m = res['mmda_forecast'][-1]
        mmda_eq = res['mmda_equilibrium']
        
        # Pass-through at various horizons (vs. base)
        if bps != 0:
            ff_chg = bps / 100.0
            pt_1m = (mmda_1m - base['mmda_forecast'][0]) / ff_chg
            pt_6m = (mmda_6m - base['mmda_forecast'][5]) / ff_chg
            pt_12m = (mmda_12m - base['mmda_forecast'][11]) / ff_chg
            pt_36m = (mmda_36m - base['mmda_forecast'][-1]) / ff_chg
            pt_eq = (mmda_eq - base['mmda_equilibrium']) / ff_chg
        else:
            pt_1m = pt_6m = pt_12m = pt_36m = pt_eq = np.nan
        
        rows.append({
            'Shock (bps)': bps,
            'FF Rate (%)': res['r_new'],
            'MMDA 1m (%)': mmda_1m,
            'MMDA 6m (%)': mmda_6m,
            'MMDA 12m (%)': mmda_12m,
            'MMDA 36m (%)': mmda_36m,
            'MMDA Equil (%)': mmda_eq,
            'Beta Start': res['beta_start'],
            'Beta Target': res['beta_target'],
            'Beta 1m': res['beta_smoothed'][0],
            'Beta 12m': res['beta_smoothed'][11],
            'Beta 36m': res['beta_smoothed'][-1],
            'PT 1m': pt_1m,
            'PT 6m': pt_6m,
            'PT 12m': pt_12m,
            'PT 36m': pt_36m,
            'PT Equil': pt_eq,
        })
    
    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUTPUT_DIR, "scenario_shock_data.csv")
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"  Saved: {csv_path}")
    return df


def print_asymmetry_summary(results):
    """Print key asymmetry findings to console."""
    base = results[0]
    print("\n" + "=" * 70)
    print("SCENARIO ANALYSIS: PARALLEL RATE SHOCK ASYMMETRY")
    print("=" * 70)
    print(f"Starting point: FF = {LAST_FF:.2f}%, MMDA = {LAST_MMDA:.3f}%")
    print(f"Model: Dynamic S-Curve (λ_up={PARAMS['lambda_up']:.3f}, λ_down={PARAMS['lambda_down']:.3f})")
    print()
    
    header = f"{'Shock':>8s}  {'FF':>6s}  {'MMDA 1m':>8s}  {'MMDA 12m':>8s}  {'MMDA 36m':>8s}  {'β Tgt':>6s}  {'PT 12m':>7s}  {'PT 36m':>7s}"
    print(header)
    print("-" * len(header))
    
    for bps in SHOCKS_BPS:
        res = results[bps]
        ff_chg = bps / 100.0 if bps != 0 else np.nan
        pt_12 = (res['mmda_forecast'][11] - base['mmda_forecast'][11]) / ff_chg if bps != 0 else np.nan
        pt_36 = (res['mmda_forecast'][-1] - base['mmda_forecast'][-1]) / ff_chg if bps != 0 else np.nan
        
        shock_str = f"+{bps}" if bps > 0 else f"{bps}" if bps < 0 else "Base"
        print(f"{shock_str:>8s}  {res['r_new']:>5.2f}%  {res['mmda_forecast'][0]:>7.3f}%  "
              f"{res['mmda_forecast'][11]:>7.3f}%  {res['mmda_forecast'][-1]:>7.3f}%  "
              f"{res['beta_target']:>.3f}  {pt_12:>6.1f}%  {pt_36:>6.1f}%")
    
    print()
    # Highlight asymmetry
    for mag in [100, 200, 300]:
        up = results[mag]
        dn = results[-mag]
        up_chg_12 = up['mmda_forecast'][11] - base['mmda_forecast'][11]
        dn_chg_12 = base['mmda_forecast'][11] - dn['mmda_forecast'][11]
        up_chg_36 = up['mmda_forecast'][-1] - base['mmda_forecast'][-1]
        dn_chg_36 = base['mmda_forecast'][-1] - dn['mmda_forecast'][-1]
        
        print(f"±{mag} bps asymmetry:")
        print(f"  12m: Up → MMDA +{up_chg_12*100:.1f} bps, Down → MMDA -{dn_chg_12*100:.1f} bps "
              f"(ratio: {up_chg_12/dn_chg_12:.2f})" if dn_chg_12 > 0 else "")
        print(f"  36m: Up → MMDA +{up_chg_36*100:.1f} bps, Down → MMDA -{dn_chg_36*100:.1f} bps "
              f"(ratio: {up_chg_36/dn_chg_36:.2f})" if dn_chg_36 > 0 else "")
    print()
    
    # S-curve driven asymmetry
    print("S-CURVE DRIVEN ASYMMETRY (equilibrium betas):")
    for bps in SHOCKS_BPS:
        if bps == 0:
            continue
        res = results[bps]
        sign = "+" if bps > 0 else ""
        print(f"  {sign}{bps} bps → FF={res['r_new']:.2f}%, β_eq={res['beta_target']:.3f}")
    print(f"  Base      → FF={LAST_FF:.2f}%, β_eq={base['beta_target']:.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Running parallel shock scenario analysis...")
    print(f"Output directory: {OUTPUT_DIR}")
    
    results = run_all_scenarios()
    
    # Generate figures
    fig_shock_paths(results,
                    os.path.join(OUTPUT_DIR, "fig_scenario_shock_paths.png"))
    fig_asymmetry(results,
                  os.path.join(OUTPUT_DIR, "fig_scenario_asymmetry.png"))
    fig_beta_convergence(results,
                         os.path.join(OUTPUT_DIR, "fig_scenario_beta_convergence.png"))
    
    # Summary data
    summary_df = generate_summary_table(results)
    
    # Console output
    print_asymmetry_summary(results)
    
    print("\nAll scenario analysis outputs saved to:", OUTPUT_DIR)
