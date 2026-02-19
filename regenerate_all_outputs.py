"""
Regenerate All Model Outputs: Enhanced Model v2.0 vs Benchmarks
================================================================

This script produces a comprehensive comparison of:
  1. Static OLS (constant beta benchmark)
  2. Linear ECM (dynamic linear benchmark)
  3. 2-Regime Nonlinear S-curve + ECM (previous recommended model)
  4. Enhanced Dynamic Beta v2.0 (AR-smoothed + sandwich SEs)

Outputs:
  - outputs/v2_comparison/fig1_model_fit_comparison.png
  - outputs/v2_comparison/fig2_scenario_forecasts.png
  - outputs/v2_comparison/fig3_beta_curves_comparison.png
  - outputs/v2_comparison/fig4_ar_smoothing_effect.png
  - outputs/v2_comparison/fig5_standard_errors_comparison.png
  - outputs/v2_comparison/fig6_residual_diagnostics.png
  - outputs/v2_comparison/fig7_sensitivity_k.png
  - outputs/v2_comparison/model_comparison_summary.csv
  - outputs/v2_comparison/parameter_estimates.csv
  - outputs/v2_comparison/sandwich_se_results.csv
  - outputs/v2_comparison/sensitivity_analysis_k.csv
  - outputs/v2_comparison/scenario_forecasts.csv
  - outputs/v2_comparison/beta_schedule.csv
  - outputs/v2_comparison/diagnostics_report.txt
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import minimize
from scipy import stats

import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf

warnings.filterwarnings("ignore")

# Import the enhanced model
from enhanced_dynamic_beta_model import (
    EnhancedDynamicBetaModel,
    logistic_beta,
    asymmetric_volatility_beta,
    ar_smoothed_beta,
)

# Import 2-regime infrastructure
from two_regime_ecm_vs_ml import (
    REGIMES, START_MIN, END_MAX, TEST_MONTHS,
    TERM_SPREAD_CANDIDATES, FHLB_SPREAD_CANDIDATES,
    coerce_numeric, beta_gompertz, choose_window,
    fit_long_run, fit_ecm, forecast_ecm,
)

# ─── Configuration ───────────────────────────────────────────────────────────

OUTPUT_DIR = "outputs/v2_comparison"
DATA_FILE = "bankratemma.csv"
FORECAST_MONTHS = 36
K_DEFAULT = 0.02  # AR smoothing tolerance

plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'figure.facecolor': 'white',
    'figure.dpi': 150,
})

COLORS = {
    'ols': '#4472C4',
    'lecm': '#70AD47',
    '2r_ecm': '#C00000',
    'v2': '#7030A0',
    'v2_unc': '#FF8C00',
    'actual': 'black',
}


# ─── Utility ─────────────────────────────────────────────────────────────────

def calc_rmse(a, b):
    a, b = np.asarray(a).flatten(), np.asarray(b).flatten()
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[m] - b[m])**2)))

def calc_mae(a, b):
    a, b = np.asarray(a).flatten(), np.asarray(b).flatten()
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.mean(np.abs(a[m] - b[m])))

def calc_r2(y, yhat):
    y, yhat = np.asarray(y).flatten(), np.asarray(yhat).flatten()
    m = np.isfinite(y) & np.isfinite(yhat)
    ss_res = np.sum((y[m] - yhat[m])**2)
    ss_tot = np.sum((y[m] - y[m].mean())**2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

def calc_dw(resid):
    r = np.asarray(resid).flatten()
    r = r[np.isfinite(r)]
    return float(np.sum(np.diff(r)**2) / np.sum(r**2)) if len(r) > 1 else np.nan

def calc_aic(n, k, ss_res):
    return n * np.log(ss_res / n) + 2 * k

def calc_bic(n, k, ss_res):
    return n * np.log(ss_res / n) + k * np.log(n)


# ─── Data Loading ────────────────────────────────────────────────────────────

def load_data_2regime():
    """Load data for 2-regime model (uses Rate/Rate_Paid convention)."""
    raw = pd.read_csv(DATA_FILE)
    raw.rename(columns={"EOM_Dt": "Date", "FEDL01": "Rate", "ILMDHYLD": "Rate_Paid"}, inplace=True)
    raw["Date"] = pd.to_datetime(raw["Date"])
    raw = raw.set_index("Date").sort_index()
    raw = raw.loc[(raw.index >= START_MIN) & (raw.index <= END_MAX)].copy()
    for c in ["Rate", "Rate_Paid"] + TERM_SPREAD_CANDIDATES + FHLB_SPREAD_CANDIDATES:
        if c in raw.columns:
            raw[c] = coerce_numeric(raw[c])
    raw = raw.dropna(subset=["Rate", "Rate_Paid"]).copy()
    return raw


# ─── Benchmark 1: Static OLS ────────────────────────────────────────────────

def fit_ols(df):
    X = sm.add_constant(df["Rate"])
    return sm.OLS(df["Rate_Paid"], X).fit(cov_type='HAC', cov_kwds={'maxlags': 6})

def predict_ols(model, df):
    X = sm.add_constant(df["Rate"])
    return pd.Series(model.predict(X), index=df.index)

def scenario_ols(model, last_rate, shock_bps, n):
    r = last_rate + shock_bps / 100.0
    return np.full(n, float(model.predict(np.array([[1.0, r]]))[0]))


# ─── Benchmark 2: Linear ECM ────────────────────────────────────────────────

def fit_lecm(df):
    tmp = df.dropna(subset=["Rate", "Rate_Paid"]).copy()
    X = sm.add_constant(tmp["Rate"])
    lr = sm.OLS(tmp["Rate_Paid"], X).fit()
    a, b = float(lr.params["const"]), float(lr.params["Rate"])
    y_star = a + b * tmp["Rate"]
    e = tmp["Rate_Paid"] - y_star

    reg = pd.DataFrame(index=tmp.index)
    reg["dY"] = tmp["Rate_Paid"].diff()
    reg["ECT_L1"] = e.shift(1)
    reg["dR"] = tmp["Rate"].diff()
    reg = reg.dropna()

    X_ecm = sm.add_constant(reg[["ECT_L1", "dR"]])
    ecm = sm.OLS(reg["dY"], X_ecm).fit(cov_type='HAC', cov_kwds={'maxlags': 6})
    return {"alpha": a, "beta": b, "lr_model": lr, "ecm_model": ecm, "residuals_lr": e}

def predict_lecm(params, df, recursive=True):
    tmp = df.dropna(subset=["Rate", "Rate_Paid"]).copy()
    a, b, ecm = params["alpha"], params["beta"], params["ecm_model"]
    dR = tmp["Rate"].diff()
    yhat = pd.Series(index=tmp.index, dtype=float)
    yhat.iloc[0] = tmp["Rate_Paid"].iloc[0]
    for i in range(1, len(tmp)):
        y_prev = float(yhat.iloc[i-1]) if recursive else float(tmp["Rate_Paid"].iloc[i-1])
        ect = y_prev - (a + b * tmp.loc[tmp.index[i-1], "Rate"])
        dr = float(dR.iloc[i])
        X = pd.DataFrame([[1.0, ect, dr]], columns=["const", "ECT_L1", "dR"])
        yhat.iloc[i] = y_prev + float(ecm.predict(X)[0])
    return yhat

def scenario_lecm(params, df, shock_bps, n):
    a, b, ecm = params["alpha"], params["beta"], params["ecm_model"]
    y0 = float(df["Rate_Paid"].iloc[-1])
    r0 = float(df["Rate"].iloc[-1])
    r_new = r0 + shock_bps / 100.0
    yhat = np.zeros(n)
    y_prev = y0
    for t in range(n):
        ect = y_prev - (a + b * (r0 if t == 0 else r_new))
        dr = (r_new - r0) if t == 0 else 0.0
        X = pd.DataFrame([[1.0, ect, dr]], columns=["const", "ECT_L1", "dR"])
        y_prev += float(ecm.predict(X)[0])
        yhat[t] = y_prev
    return yhat


# ─── Benchmark 3: 2-Regime S-curve ECM ──────────────────────────────────────

def fit_2regime(raw):
    bundles = []
    for name, cfg in REGIMES.items():
        best = choose_window(raw, cfg["start_candidates"], cfg["end_candidates"], int(cfg["min_obs"]))
        df_win = raw.loc[(raw.index >= best.start) & (raw.index <= best.end)].copy()
        df_win = df_win.dropna(subset=["Rate", "Rate_Paid", best.term_col, best.fhlb_col]).copy()
        lr_fit = fit_long_run(df_win, best.term_col, best.fhlb_col)
        if lr_fit is None:
            raise RuntimeError(f"Long-run fit failed for {name}")
        lr_params, y_star, e, _, _ = lr_fit
        ecm_res, ecm_cols = fit_ecm(df_win, best.term_col, best.fhlb_col, lr_params, asymmetric=False)
        ecm_s, ecm_cols_s = fit_ecm(df_win, best.term_col, best.fhlb_col, lr_params, asymmetric=False, max_p=1, max_q=1)
        bundles.append({"name": name, "start": best.start, "end": best.end,
                       "term_col": best.term_col, "fhlb_col": best.fhlb_col,
                       "lr_params": lr_params, "y_star": y_star, "e": e,
                       "ecm_res": ecm_res, "ecm_cols": ecm_cols,
                       "ecm_s": ecm_s, "ecm_cols_s": ecm_cols_s, "df_win": df_win})
    return bundles

def predict_2regime(bundles):
    results = []
    for b in bundles:
        df_win = b["df_win"]
        y_ecm = forecast_ecm(df_win.iloc[:1], df_win.iloc[1:], b["term_col"], b["fhlb_col"],
                              b["lr_params"], b["ecm_res"], b["ecm_cols"], asymmetric=False, recursive=True)
        y_full = pd.concat([df_win["Rate_Paid"].iloc[:1], y_ecm])
        results.append(pd.DataFrame({"Date": df_win.index, "Actual": df_win["Rate_Paid"].values,
                                     "Fitted": y_full.reindex(df_win.index).values,
                                     "Rate": df_win["Rate"].values, "Regime": b["name"]}))
    return pd.concat(results, ignore_index=True)

def scenario_2regime(bundles, shock_bps, n):
    b = bundles[-1]
    df_win = b["df_win"]
    future = pd.date_range(start=df_win.index[-1] + pd.DateOffset(months=1), periods=n, freq="ME")
    df_fut = pd.DataFrame(index=future)
    df_fut["Rate"] = float(df_win["Rate"].iloc[-1]) + shock_bps / 100.0
    df_fut[b["term_col"]] = float(df_win[b["term_col"]].iloc[-1])
    df_fut[b["fhlb_col"]] = float(df_win[b["fhlb_col"]].iloc[-1])
    df_fut["Rate_Paid"] = np.nan
    return forecast_ecm(df_win, df_fut, b["term_col"], b["fhlb_col"], b["lr_params"],
                        b["ecm_s"], b["ecm_cols_s"], asymmetric=False, recursive=True).values


# ─── Enhanced Model v2.0 Scenarios ──────────────────────────────────────────

def scenario_v2(model, shock_bps, n_months, fhlb_last=None):
    """Generate scenario forecast for Enhanced v2.0 model."""
    p = model.params
    last_rate = model.FEDL01[-1]
    last_mmda = model.Y[-1]
    last_fhlb = model.FHLK3MSPRD[-1] if fhlb_last is None else fhlb_last
    last_beta_smooth = model.beta_smoothed[-1]
    
    r_new = last_rate + shock_bps / 100.0
    
    # Target beta at new rate (no volatility adjustment in scenario, vol_ratio=1)
    beta_eq = logistic_beta(np.array([r_new]), p['k'], p['m'], p['beta_min'], p['beta_max'])[0]
    target = p['alpha'] + beta_eq * r_new + p['gamma_fhlb'] * last_fhlb
    
    # Simulate with AR-smoothed beta convergence
    yhat = np.zeros(n_months)
    beta_prev = last_beta_smooth
    y_prev = last_mmda
    
    for t in range(n_months):
        # Beta converges to equilibrium with AR constraint
        delta = beta_eq - beta_prev
        clamped = np.clip(delta, -model.max_beta_change, model.max_beta_change)
        beta_t = beta_prev + clamped
        
        # Prediction at this period
        yhat[t] = p['alpha'] + beta_t * r_new + p['gamma_fhlb'] * last_fhlb
        beta_prev = beta_t
    
    return yhat


# ═══════════════════════════════════════════════════════════════════════════════
# CHART GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def fig1_model_fit(df, y_ols, y_lecm, regime_df, v2_dates, v2_pred, v2_actual, path):
    """Figure 1: 4-model fit comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    
    # Panel A: Full sample time series
    ax = axes[0, 0]
    ax.plot(df.index, df["Rate_Paid"], 'k-', lw=1.5, label='Actual MMDA')
    ax.plot(df.index, y_ols, color=COLORS['ols'], ls='--', lw=1, alpha=0.7, label='Static OLS')
    ax.plot(df.index, y_lecm, color=COLORS['lecm'], ls='-.', lw=1, alpha=0.7, label='Linear ECM')
    for regime in regime_df["Regime"].unique():
        mask = regime_df["Regime"] == regime
        dates_r = pd.to_datetime(regime_df.loc[mask, "Date"])
        ax.plot(dates_r, regime_df.loc[mask, "Fitted"], color=COLORS['2r_ecm'], lw=1.2,
                label='2-Regime ECM' if regime == regime_df["Regime"].unique()[0] else None)
    ax.plot(v2_dates, v2_pred, color=COLORS['v2'], lw=1.8, label='Enhanced v2.0')
    ax.set_ylabel("MMDA Rate (%)")
    ax.set_title("(A) Full-Sample Model Fit", fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.grid(True, alpha=0.3)
    
    # Panel B: 2022-2025 zoom
    ax = axes[0, 1]
    zoom_start = "2022-01-01"
    z = df.index >= zoom_start
    ax.plot(df.index[z], df["Rate_Paid"][z], 'ko', ms=4, alpha=0.6, label='Actual')
    ax.plot(df.index[z], y_ols[z], color=COLORS['ols'], ls='--', lw=1, label='OLS')
    ax.plot(df.index[z], y_lecm[z], color=COLORS['lecm'], ls='-.', lw=1, label='LECM')
    v2_z = v2_dates >= pd.Timestamp(zoom_start)
    ax.plot(v2_dates[v2_z], v2_pred[v2_z], color=COLORS['v2'], lw=2, label='v2.0')
    ax.set_ylabel("MMDA Rate (%)")
    ax.set_title("(B) Hiking Cycle Zoom (2022-2025)", fontweight='bold')
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.grid(True, alpha=0.3)
    
    # Panel C: R² bar chart
    ax = axes[1, 0]
    r2_ols = calc_r2(df["Rate_Paid"], y_ols)
    r2_lecm = calc_r2(df["Rate_Paid"], y_lecm)
    r2_2r = calc_r2(regime_df["Actual"], regime_df["Fitted"])
    r2_v2 = calc_r2(v2_actual, v2_pred)
    
    labels = ['Static OLS', 'Linear ECM', '2-Regime\nS-curve ECM', 'Enhanced\nv2.0']
    r2_vals = [r2_ols, r2_lecm, r2_2r, r2_v2]
    colors_bar = [COLORS['ols'], COLORS['lecm'], COLORS['2r_ecm'], COLORS['v2']]
    bars = ax.bar(labels, r2_vals, color=colors_bar, edgecolor='black', lw=0.5)
    ax.set_ylabel("R²")
    ax.set_title("(C) In-Sample R² Comparison", fontweight='bold')
    ax.set_ylim(0.90, 1.0)
    for bar, val in zip(bars, r2_vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.002, f'{val:.4f}', ha='center', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel D: RMSE bar chart
    ax = axes[1, 1]
    rmse_vals = [calc_rmse(df["Rate_Paid"], y_ols) * 100,
                 calc_rmse(df["Rate_Paid"], y_lecm) * 100,
                 calc_rmse(regime_df["Actual"], regime_df["Fitted"]) * 100,
                 calc_rmse(v2_actual, v2_pred) * 100]
    bars = ax.bar(labels, rmse_vals, color=colors_bar, edgecolor='black', lw=0.5)
    ax.set_ylabel("RMSE (bps)")
    ax.set_title("(D) In-Sample RMSE Comparison", fontweight='bold')
    for bar, val in zip(bars, rmse_vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.3, f'{val:.1f}', ha='center', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def fig2_scenarios(df, bundles, lecm_params, ols_model, v2_model, path):
    """Figure 2: Scenario forecasts across all 4 models."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    
    last_date = df.index[-1]
    fcast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1),
                                periods=FORECAST_MONTHS, freq="ME")
    
    shocks = [("Base (Flat)", 0), ("+200 bps", 200), ("-100 bps", -100)]
    scen = {}
    for name, bps in shocks:
        scen[name] = {
            "OLS": scenario_ols(ols_model, float(df["Rate"].iloc[-1]), bps, FORECAST_MONTHS),
            "LECM": scenario_lecm(lecm_params, df, bps, FORECAST_MONTHS),
            "2R-ECM": scenario_2regime(bundles, bps, FORECAST_MONTHS),
            "v2.0": scenario_v2(v2_model, bps, FORECAST_MONTHS),
        }
    
    hist = df.index[-24:]
    hist_y = df.loc[hist, "Rate_Paid"]
    scen_colors = {'Base (Flat)': 'blue', '+200 bps': 'red', '-100 bps': 'green'}
    
    panels = [("OLS", "(A) Static OLS"), ("LECM", "(B) Linear ECM"),
              ("2R-ECM", "(C) 2-Regime S-curve ECM"), ("v2.0", "(D) Enhanced v2.0")]
    
    for idx, (key, title) in enumerate(panels):
        ax = axes[idx // 2, idx % 2]
        ax.plot(hist, hist_y, 'k-', lw=1.5, label='Historical')
        ax.axvline(last_date, color='gray', ls='--', alpha=0.5)
        for sn, sd in scen.items():
            ls = '-' if key == 'v2.0' else '--'
            ax.plot(fcast_dates, sd[key], color=scen_colors[sn], ls=ls, lw=1.3, label=sn)
        ax.set_xlabel("Date")
        ax.set_ylabel("MMDA Rate (%)")
        ax.set_title(title, fontweight='bold')
        ax.legend(loc='upper left', fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    
    # Save scenario data
    sdf = pd.DataFrame({"Date": fcast_dates})
    for sn in scen:
        for mk in scen[sn]:
            sdf[f"{mk}_{sn}"] = scen[sn][mk]
    sdf.to_csv(os.path.join(OUTPUT_DIR, "scenario_forecasts.csv"), index=False)
    return scen


def fig3_beta_curves(v2_model, bundles, ols_model, path):
    """Figure 3: Beta curves from all model families."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    p = v2_model.params
    r_grid = np.linspace(0, 10, 200)
    
    # Panel A: Beta curves
    ax = axes[0]
    # Static OLS
    ols_beta = float(ols_model.params["Rate"])
    ax.axhline(ols_beta, color=COLORS['ols'], ls='--', lw=1.5, label=f'Static OLS (β={ols_beta:.3f})')
    
    # 2-Regime betas
    for i, b in enumerate(bundles):
        lr = b["lr_params"]
        beta_vals = [beta_gompertz(r, lr.k, lr.m, lr.beta_min, lr.beta_max) for r in r_grid]
        ax.plot(r_grid, beta_vals, color=COLORS['2r_ecm'], ls=('-' if i == 0 else ':'), lw=1.5,
                label=f'2R-ECM: {b["name"]}', alpha=0.8)
    
    # Enhanced v2.0
    beta_v2 = logistic_beta(r_grid, p['k'], p['m'], p['beta_min'], p['beta_max'])
    ax.plot(r_grid, beta_v2, color=COLORS['v2'], lw=2.5, label='Enhanced v2.0 (S-curve)')
    
    ax.set_xlabel("Fed Funds Rate (%)")
    ax.set_ylabel("Deposit Beta")
    ax.set_title("(A) Equilibrium Beta by Rate Level", fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 10)
    ax.set_ylim(0.2, 0.85)
    
    # Panel B: Beta schedule table
    ax = axes[1]
    ax.axis('off')
    
    rates = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    rows = [["Rate", "OLS", "v2.0 S-curve"]]
    if len(bundles) >= 2:
        rows[0].extend(["Regime 1", "Regime 2"])
    for r in rates:
        row = [f"{r}%", f"{ols_beta*100:.1f}%",
               f"{logistic_beta(np.array([r]), p['k'], p['m'], p['beta_min'], p['beta_max'])[0]*100:.1f}%"]
        for b in bundles:
            lr = b["lr_params"]
            row.append(f"{beta_gompertz(r, lr.k, lr.m, lr.beta_min, lr.beta_max)*100:.1f}%")
        rows.append(row)
    
    tbl = ax.table(cellText=rows, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 1.5)
    for i in range(len(rows[0])):
        tbl[(0, i)].set_facecolor('#4472C4')
        tbl[(0, i)].set_text_props(color='white', weight='bold')
    ax.set_title("(B) Deposit Beta Schedule", pad=20, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    
    # Save schedule CSV
    schedule_df = pd.DataFrame(rows[1:], columns=rows[0])
    schedule_df.to_csv(os.path.join(OUTPUT_DIR, "beta_schedule.csv"), index=False)


def fig4_ar_smoothing(v2_model, path):
    """Figure 4: AR smoothing effect on beta."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    dates = v2_model.data['EOM_Dt']
    
    # Panel A: Unconstrained vs smoothed beta
    ax = axes[0, 0]
    ax.plot(dates, v2_model.beta_unconstrained * 100, color=COLORS['v2_unc'], alpha=0.5, lw=1,
            label='Unconstrained β*')
    ax.plot(dates, v2_model.beta_smoothed * 100, color=COLORS['v2'], lw=2.5,
            label=f'AR-Smoothed β̃ (k={v2_model.max_beta_change})')
    ax.set_ylabel("Beta (%)")
    ax.set_title("(A) Beta Smoothing: Unconstrained vs AR-Constrained", fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Panel B: Period-to-period changes
    ax = axes[0, 1]
    d_unc = np.diff(v2_model.beta_unconstrained) * 100
    d_sm = np.diff(v2_model.beta_smoothed) * 100
    ax.plot(dates[1:], d_unc, color=COLORS['v2_unc'], alpha=0.4, lw=1, label='Δβ unconstrained')
    ax.plot(dates[1:], d_sm, color=COLORS['v2'], lw=2, label='Δβ smoothed')
    ax.axhline(v2_model.max_beta_change * 100, color='green', ls='--', alpha=0.7,
               label=f'±k = ±{v2_model.max_beta_change*100:.0f} pp')
    ax.axhline(-v2_model.max_beta_change * 100, color='green', ls='--', alpha=0.7)
    ax.set_ylabel("Δβ (pp/month)")
    ax.set_title("(B) Period-to-Period Beta Changes", fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Panel C: Histogram of beta changes
    ax = axes[1, 0]
    ax.hist(d_unc, bins=25, alpha=0.5, color=COLORS['v2_unc'], label='Unconstrained', density=True)
    ax.hist(d_sm, bins=25, alpha=0.7, color=COLORS['v2'], label='AR-Smoothed', density=True)
    ax.axvline(v2_model.max_beta_change * 100, color='green', ls='--')
    ax.axvline(-v2_model.max_beta_change * 100, color='green', ls='--')
    ax.set_xlabel("Δβ (pp/month)")
    ax.set_ylabel("Density")
    ax.set_title("(C) Distribution of Beta Changes", fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Panel D: Beta with Fed Funds overlay
    ax = axes[1, 1]
    ax.plot(dates, v2_model.beta_smoothed * 100, color=COLORS['v2'], lw=2.5, label='AR-Smoothed β̃')
    ax.set_ylabel("Beta (%)", color=COLORS['v2'])
    ax.tick_params(axis='y', labelcolor=COLORS['v2'])
    ax2 = ax.twinx()
    ax2.fill_between(dates, 0, v2_model.FEDL01, alpha=0.15, color='gray')
    ax2.set_ylabel("Fed Funds Rate (%)", color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    ax.set_title("(D) Dynamic Beta vs Rate Environment", fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def fig5_standard_errors(se_results, path):
    """Figure 5: Sandwich vs naive standard errors."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    tbl = se_results['param_table']
    
    # Panel A: SE comparison bars
    ax = axes[0]
    x = np.arange(len(tbl))
    w = 0.35
    ax.bar(x - w/2, tbl['Naive SE'], w, label='Naive (Hessian)', color='skyblue', edgecolor='black', lw=0.5)
    ax.bar(x + w/2, tbl['Sandwich SE'], w, label='Sandwich (HAC)', color='coral', edgecolor='black', lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(tbl['Parameter'], rotation=45, ha='right')
    ax.set_ylabel("Standard Error")
    ax.set_title("(A) Naive vs Sandwich SE", fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel B: SE ratio
    ax = axes[1]
    ratios = tbl['SE Ratio'].values
    colors_ratio = ['coral' if r > 1.5 else 'gold' if r > 1 else 'lightgreen' for r in ratios]
    bars = ax.bar(x, ratios, color=colors_ratio, edgecolor='black', lw=0.5)
    ax.axhline(1.0, color='black', ls='--', lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(tbl['Parameter'], rotation=45, ha='right')
    ax.set_ylabel("Sandwich SE / Naive SE")
    ax.set_title("(B) SE Inflation Ratio", fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel C: t-statistics
    ax = axes[2]
    t_vals = np.abs(tbl['t-stat'].values)
    t_colors = ['#70AD47' if t > 2 else '#C00000' for t in t_vals]
    ax.barh(x, t_vals, color=t_colors, edgecolor='black', lw=0.5)
    ax.axvline(1.96, color='red', ls='--', lw=1, label='5% significance')
    ax.set_yticks(x)
    ax.set_yticklabels(tbl['Parameter'])
    ax.set_xlabel("|t-statistic|")
    ax.set_title("(C) Statistical Significance (Sandwich)", fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def fig6_residual_diagnostics(v2_model, y_ols, y_lecm, df, path):
    """Figure 6: Residual diagnostics comparison."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    dates = v2_model.data['EOM_Dt']
    
    resid_ols = (df["Rate_Paid"] - y_ols).values
    resid_lecm = (df["Rate_Paid"] - y_lecm).dropna().values
    resid_v2 = v2_model.residuals
    
    models_resid = [("Static OLS", resid_ols, COLORS['ols']),
                    ("Linear ECM", resid_lecm, COLORS['lecm']),
                    ("Enhanced v2.0", resid_v2, COLORS['v2'])]
    
    # Row 1: Time series of residuals
    for i, (name, resid, color) in enumerate(models_resid):
        ax = axes[0, i]
        ax.bar(range(len(resid)), resid * 100, alpha=0.7, color=color, width=1.0)
        ax.axhline(0, color='black', lw=0.5)
        rmse_val = np.sqrt(np.mean(resid**2)) * 100
        ax.set_title(f"(A{i+1}) {name} Residuals\nRMSE={rmse_val:.1f} bps", fontweight='bold')
        ax.set_ylabel("Residual (bps)")
        ax.grid(True, alpha=0.3)
    
    # Row 2: ACF of residuals
    for i, (name, resid, color) in enumerate(models_resid):
        ax = axes[1, i]
        clean = resid[np.isfinite(resid)]
        plot_acf(clean, ax=ax, lags=15, alpha=0.05)
        ax.set_title(f"(B{i+1}) {name} ACF", fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def fig7_sensitivity_k(v2_model, path):
    """Figure 7: Sensitivity analysis on smoothing parameter k."""
    k_values = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.05, 0.10, np.inf]
    params = v2_model.result.x
    Y = v2_model.Y
    FEDL01 = v2_model.FEDL01
    FHLK3MSPRD = v2_model.FHLK3MSPRD
    
    rows = []
    for kv in k_values:
        beta_s = ar_smoothed_beta(v2_model.beta_unconstrained, kv)
        pred = params[0] + beta_s * FEDL01 + params[5] * FHLK3MSPRD
        resid = Y - pred
        ss_res = np.sum(resid**2)
        ss_tot = np.sum((Y - Y.mean())**2)
        changes = np.abs(np.diff(beta_s))
        rows.append({
            'k': kv if np.isfinite(kv) else 999,
            'k_label': f'{kv:.3f}' if np.isfinite(kv) else '∞',
            'r2': 1 - ss_res / ss_tot,
            'rmse': np.sqrt(np.mean(resid**2)),
            'max_dbeta': changes.max(),
            'mean_dbeta': changes.mean(),
            'n_constrained': int(np.sum(np.abs(np.diff(v2_model.beta_unconstrained)) > kv)),
        })
    
    df_k = pd.DataFrame(rows)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Panel A: R² vs k
    ax = axes[0]
    ax.plot(range(len(df_k)), df_k['r2'], 'bo-', lw=2, ms=8)
    ax.set_xticks(range(len(df_k)))
    ax.set_xticklabels(df_k['k_label'], rotation=45)
    ax.set_xlabel("Smoothing Constraint k")
    ax.set_ylabel("R²")
    ax.set_title("(A) Model Fit vs Smoothing", fontweight='bold')
    ax.axvline(3, color='red', ls='--', alpha=0.5, label='k=0.02 (default)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel B: Max beta change vs k
    ax = axes[1]
    ax.plot(range(len(df_k)), df_k['max_dbeta'] * 100, 'rs-', lw=2, ms=8)
    ax.set_xticks(range(len(df_k)))
    ax.set_xticklabels(df_k['k_label'], rotation=45)
    ax.set_xlabel("Smoothing Constraint k")
    ax.set_ylabel("Max |Δβ| (pp)")
    ax.set_title("(B) Beta Volatility vs Smoothing", fontweight='bold')
    ax.axvline(3, color='red', ls='--', alpha=0.5, label='k=0.02 (default)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel C: Trade-off scatter
    ax = axes[2]
    ax.scatter(df_k['max_dbeta'] * 100, df_k['r2'], s=100, c='purple', zorder=5)
    for _, row in df_k.iterrows():
        ax.annotate(row['k_label'], (row['max_dbeta'] * 100, row['r2']),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("Max |Δβ| (pp)")
    ax.set_ylabel("R²")
    ax.set_title("(C) Fit-Stability Trade-off", fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    
    # Save sensitivity CSV
    df_k_out = df_k.copy()
    df_k_out['k'] = df_k_out['k_label']
    df_k_out.drop(columns=['k_label'], inplace=True)
    df_k_out.to_csv(os.path.join(OUTPUT_DIR, "sensitivity_analysis_k.csv"), index=False)


# ═══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTICS REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def generate_report(df, y_ols, y_lecm, regime_df, v2_model, se_results, bundles):
    """Generate comprehensive diagnostics text report."""
    lines = []
    lines.append("=" * 90)
    lines.append("MODEL COMPARISON REPORT: Enhanced v2.0 vs Benchmarks")
    lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("=" * 90)
    
    # ── Performance comparison
    lines.append("\n1. MODEL PERFORMANCE COMPARISON")
    lines.append("-" * 60)
    
    models_info = [
        ("Static OLS", y_ols.values, df["Rate_Paid"].values, 2),
        ("Linear ECM", predict_lecm(fit_lecm(df), df, True).values, df["Rate_Paid"].values, 4),
    ]
    
    # Format as table
    lines.append(f"  {'Model':<25} {'R²':>8} {'RMSE(bps)':>10} {'MAE(bps)':>10} {'AIC':>10} {'BIC':>10}")
    lines.append("  " + "─" * 75)
    
    n = len(df)
    for name, yhat, ytrue, k in models_info:
        r2 = calc_r2(ytrue, yhat)
        rm = calc_rmse(ytrue, yhat)
        ma = calc_mae(ytrue, yhat)
        ss = np.sum((ytrue - yhat)**2)
        aic = calc_aic(n, k, ss)
        bic = calc_bic(n, k, ss)
        lines.append(f"  {name:<25} {r2:>8.4f} {rm*100:>10.2f} {ma*100:>10.2f} {aic:>10.1f} {bic:>10.1f}")
    
    # 2-Regime
    r2_2r = calc_r2(regime_df["Actual"], regime_df["Fitted"])
    rm_2r = calc_rmse(regime_df["Actual"], regime_df["Fitted"])
    ma_2r = calc_mae(regime_df["Actual"], regime_df["Fitted"])
    lines.append(f"  {'2-Regime S-curve ECM':<25} {r2_2r:>8.4f} {rm_2r*100:>10.2f} {ma_2r*100:>10.2f} {'N/A':>10} {'N/A':>10}")
    
    # v2.0
    r2_v2 = calc_r2(v2_model.Y, v2_model.predictions)
    rm_v2 = calc_rmse(v2_model.Y, v2_model.predictions)
    ma_v2 = calc_mae(v2_model.Y, v2_model.predictions)
    ss_v2 = np.sum(v2_model.residuals**2)
    aic_v2 = calc_aic(len(v2_model.Y), 8, ss_v2)
    bic_v2 = calc_bic(len(v2_model.Y), 8, ss_v2)
    lines.append(f"  {'Enhanced v2.0 (AR-smooth)':<25} {r2_v2:>8.4f} {rm_v2*100:>10.2f} {ma_v2*100:>10.2f} {aic_v2:>10.1f} {bic_v2:>10.1f}")
    
    # ── Parameter estimates
    lines.append("\n2. ENHANCED v2.0 PARAMETER ESTIMATES (with Sandwich SEs)")
    lines.append("-" * 60)
    lines.append(se_results['param_table'].to_string(index=False))
    
    # ── Beta smoothness
    lines.append("\n3. BETA SMOOTHNESS ASSESSMENT")
    lines.append("-" * 60)
    metrics = v2_model.compute_metrics()
    lines.append(f"  Max Δβ constraint (k):      {v2_model.max_beta_change:.4f}")
    lines.append(f"  Max Δβ (unconstrained):     {metrics['max_beta_change_unconstrained']:.4f}")
    lines.append(f"  Max Δβ (smoothed):          {metrics['max_beta_change_actual']:.4f}")
    lines.append(f"  Mean Δβ (unconstrained):    {metrics['mean_beta_change_unconstrained']:.4f}")
    lines.append(f"  Mean Δβ (smoothed):         {metrics['mean_beta_change_actual']:.4f}")
    lines.append(f"  Periods constrained:        {metrics['n_constrained_periods']} / {metrics['n_obs'] - 1}")
    
    # ── Residual diagnostics
    lines.append("\n4. RESIDUAL DIAGNOSTICS")
    lines.append("-" * 60)
    
    resid_v2 = v2_model.residuals
    dw = calc_dw(resid_v2)
    clean_resid = resid_v2[np.isfinite(resid_v2)]
    
    try:
        lb = acorr_ljungbox(clean_resid, lags=[10], return_df=True)
        lb_stat, lb_pval = float(lb["lb_stat"].iloc[0]), float(lb["lb_pvalue"].iloc[0])
    except:
        lb_stat, lb_pval = np.nan, np.nan
    
    jb_stat, jb_pval = stats.jarque_bera(clean_resid)
    
    lines.append(f"  Enhanced v2.0:")
    lines.append(f"    Durbin-Watson:     {dw:.4f}")
    lines.append(f"    Ljung-Box(10):     stat={lb_stat:.4f}, p={lb_pval:.4f}")
    lines.append(f"    Jarque-Bera:       stat={jb_stat:.4f}, p={jb_pval:.4f}")
    lines.append(f"    Residual mean:     {np.mean(clean_resid)*100:.2f} bps")
    lines.append(f"    Residual std:      {np.std(clean_resid)*100:.2f} bps")
    
    # ── SE comparison note
    lines.append("\n5. STANDARD ERROR METHODOLOGY")
    lines.append("-" * 60)
    lines.append("  Naive SEs: V = H^{-1} (assumes independent observations)")
    lines.append("  Sandwich SEs: V = H^{-1} S H^{-1} (Huber-White with Newey-West HAC)")
    lines.append(f"  Newey-West lags used: {se_results['n_lags']}")
    lines.append("  SE ratio > 1 indicates naive SEs understate uncertainty")
    lines.append("  SE ratio < 1 indicates the nonlinear curvature of the likelihood")
    lines.append("    surface provides more information than the outer product of scores")
    
    lines.append("\n" + "=" * 90)
    
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 90)
    print(" REGENERATING ALL MODEL OUTPUTS")
    print(" Enhanced Dynamic Beta v2.0 vs Benchmarks")
    print("=" * 90)
    
    # ── Step 1: Load data ─────────────────────────────────────────────────────
    print("\n[1/8] Loading data...")
    raw_2r = load_data_2regime()
    df = raw_2r.copy()
    print(f"  {len(df)} observations, {df.index.min().date()} to {df.index.max().date()}")
    
    # ── Step 2: Fit Static OLS ────────────────────────────────────────────────
    print("[2/8] Fitting Static OLS benchmark...")
    ols_model = fit_ols(df)
    y_ols = predict_ols(ols_model, df)
    print(f"  OLS R² = {calc_r2(df['Rate_Paid'], y_ols):.4f}")
    
    # ── Step 3: Fit Linear ECM ────────────────────────────────────────────────
    print("[3/8] Fitting Linear ECM benchmark...")
    lecm_params = fit_lecm(df)
    y_lecm = predict_lecm(lecm_params, df, recursive=True)
    print(f"  LECM R² = {calc_r2(df['Rate_Paid'], y_lecm):.4f}")
    
    # ── Step 4: Fit 2-Regime S-curve ECM ──────────────────────────────────────
    print("[4/8] Fitting 2-Regime S-curve ECM...")
    bundles = fit_2regime(raw_2r)
    regime_df = predict_2regime(bundles)
    print(f"  2R-ECM R² = {calc_r2(regime_df['Actual'], regime_df['Fitted']):.4f}")
    
    # ── Step 5: Fit Enhanced v2.0 ─────────────────────────────────────────────
    print("[5/8] Fitting Enhanced Dynamic Beta v2.0...")
    v2 = EnhancedDynamicBetaModel(max_beta_change=K_DEFAULT)
    v2.load_data(DATA_FILE)
    v2.estimate(n_starts=5)
    se_results = v2.compute_sandwich_se()
    print(f"  v2.0 R² = {calc_r2(v2.Y, v2.predictions):.4f}, RMSE = {calc_rmse(v2.Y, v2.predictions)*100:.2f} bps")
    
    # ── Step 6: Generate all figures ──────────────────────────────────────────
    print("\n[6/8] Generating comparison figures...")
    
    v2_dates = v2.data['EOM_Dt']
    
    fig1_model_fit(df, y_ols, y_lecm, regime_df, v2_dates, v2.predictions, v2.Y,
                   os.path.join(OUTPUT_DIR, "fig1_model_fit_comparison.png"))
    
    fig2_scenarios(df, bundles, lecm_params, ols_model, v2,
                   os.path.join(OUTPUT_DIR, "fig2_scenario_forecasts.png"))
    
    fig3_beta_curves(v2, bundles, ols_model,
                     os.path.join(OUTPUT_DIR, "fig3_beta_curves_comparison.png"))
    
    fig4_ar_smoothing(v2, os.path.join(OUTPUT_DIR, "fig4_ar_smoothing_effect.png"))
    
    fig5_standard_errors(se_results,
                         os.path.join(OUTPUT_DIR, "fig5_standard_errors_comparison.png"))
    
    fig6_residual_diagnostics(v2, y_ols, y_lecm, df,
                              os.path.join(OUTPUT_DIR, "fig6_residual_diagnostics.png"))
    
    fig7_sensitivity_k(v2, os.path.join(OUTPUT_DIR, "fig7_sensitivity_k.png"))
    
    # ── Step 7: Save data outputs ─────────────────────────────────────────────
    print("\n[7/8] Saving data outputs...")
    
    # Model comparison summary
    summary_rows = []
    summary_rows.append({"Model": "Static OLS", "R2": calc_r2(df["Rate_Paid"], y_ols),
                          "RMSE_bps": calc_rmse(df["Rate_Paid"], y_ols) * 100,
                          "MAE_bps": calc_mae(df["Rate_Paid"], y_ols) * 100,
                          "Parameters": 2, "Beta_Type": "Constant"})
    summary_rows.append({"Model": "Linear ECM", "R2": calc_r2(df["Rate_Paid"], y_lecm),
                          "RMSE_bps": calc_rmse(df["Rate_Paid"], y_lecm) * 100,
                          "MAE_bps": calc_mae(df["Rate_Paid"], y_lecm) * 100,
                          "Parameters": 4, "Beta_Type": "Constant + ECM"})
    summary_rows.append({"Model": "2-Regime S-curve ECM", "R2": calc_r2(regime_df["Actual"], regime_df["Fitted"]),
                          "RMSE_bps": calc_rmse(regime_df["Actual"], regime_df["Fitted"]) * 100,
                          "MAE_bps": calc_mae(regime_df["Actual"], regime_df["Fitted"]) * 100,
                          "Parameters": "~12/regime", "Beta_Type": "Gompertz S-curve + ECM"})
    summary_rows.append({"Model": "Enhanced v2.0 (AR-smoothed)", "R2": calc_r2(v2.Y, v2.predictions),
                          "RMSE_bps": calc_rmse(v2.Y, v2.predictions) * 100,
                          "MAE_bps": calc_mae(v2.Y, v2.predictions) * 100,
                          "Parameters": 8, "Beta_Type": "Logistic S-curve + Vol + AR"})
    
    pd.DataFrame(summary_rows).to_csv(os.path.join(OUTPUT_DIR, "model_comparison_summary.csv"), index=False)
    print(f"  Saved: model_comparison_summary.csv")
    
    # Parameter estimates
    param_df = se_results['param_table'].copy()
    param_df.to_csv(os.path.join(OUTPUT_DIR, "parameter_estimates.csv"), index=False)
    print(f"  Saved: parameter_estimates.csv")
    
    # Sandwich SE results
    se_detail = pd.DataFrame({
        'Parameter': se_results['param_names'],
        'Estimate': v2.result.x,
        'Naive_SE': se_results['hessian_se'],
        'Sandwich_SE': se_results['sandwich_se'],
        'SE_Ratio': se_results['sandwich_se'] / np.where(se_results['hessian_se'] > 0, se_results['hessian_se'], np.nan),
        't_stat': se_results['t_statistics'],
        'p_value': se_results['p_values'],
    })
    se_detail.to_csv(os.path.join(OUTPUT_DIR, "sandwich_se_results.csv"), index=False)
    print(f"  Saved: sandwich_se_results.csv")
    
    # ── Step 8: Generate diagnostics report ───────────────────────────────────
    print("\n[8/8] Generating diagnostics report...")
    report = generate_report(df, y_ols, y_lecm, regime_df, v2, se_results, bundles)
    with open(os.path.join(OUTPUT_DIR, "diagnostics_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Saved: diagnostics_report.txt")
    
    # Print to console
    print("\n" + report)
    
    # ── Also update the original outputs directory ────────────────────────────
    print("\n" + "=" * 90)
    print(" UPDATING ORIGINAL OUTPUT DIRECTORIES")
    print("=" * 90)
    
    # Re-run the enhanced model standalone outputs
    v2.print_results()
    v2.compare_with_unconstrained()
    v2.sensitivity_analysis_k()
    v2.plot_results(save_path='outputs/figures/enhanced_model_v2.png')
    
    print("\n" + "=" * 90)
    print(" ALL OUTPUTS REGENERATED SUCCESSFULLY")
    print("=" * 90)
    print(f"\n Output directory: {OUTPUT_DIR}/")
    print(f" Files generated:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, f)
        size = os.path.getsize(fpath)
        print(f"   {f:<45} {size:>10,} bytes")


if __name__ == "__main__":
    main()
