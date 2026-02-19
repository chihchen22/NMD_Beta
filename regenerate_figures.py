"""
Regenerate all figures for the research paper with updated model parameters.
Updated model: 8 parameters (without term spread)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import minimize
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['figure.dpi'] = 150

# Create output directories
os.makedirs('outputs/visualizations', exist_ok=True)
os.makedirs('outputs/figures', exist_ok=True)

# Load data
print("Loading data...")
df = pd.read_csv('bankratemma.csv')
df['EOM_Dt'] = pd.to_datetime(df['EOM_Dt'])

# Filter date range
df = df[(df['EOM_Dt'] >= '2017-01-01') & (df['EOM_Dt'] <= '2025-03-31')].copy()

# Clean FHLB spread
df['FHLK3MSPRD'] = pd.to_numeric(df['FHLK3MSPRD'], errors='coerce')

# Calculate volatility
df['FEDL01_change'] = df['FEDL01'].diff()
df['rolling_vol'] = df['FEDL01_change'].rolling(window=24, min_periods=12).std()
df['expanding_vol_mean'] = df['rolling_vol'].expanding(min_periods=12).mean()
df['vol_ratio'] = df['rolling_vol'] / df['expanding_vol_mean']
df = df.dropna(subset=['vol_ratio', 'FHLK3MSPRD']).reset_index(drop=True)

print(f"Data loaded: {len(df)} observations")

# Updated model parameters (8-parameter model without term spread)
PARAMS = {
    'alpha': -0.0186,
    'k': 0.5663,
    'm': 2.6550,
    'beta_min': 0.4000,
    'beta_max': 0.7000,
    'gamma_fhlb': 1.3551,
    'lambda_up': 0.2560,
    'lambda_down': 0.2046
}

# Static OLS parameters
STATIC_ALPHA = 0.2494
STATIC_BETA = 0.4617

def logistic_beta(fed_funds, k, m, beta_min, beta_max):
    """Calculate level-dependent beta using logistic function."""
    return beta_min + (beta_max - beta_min) / (1 + np.exp(-k * (fed_funds - m)))

def asymmetric_volatility_beta(fed_funds, k, m, beta_min, beta_max, vol_ratio, 
                                lambda_up, lambda_down, rate_change):
    """Calculate asymmetric volatility-adjusted beta."""
    beta_level = logistic_beta(fed_funds, k, m, beta_min, beta_max)
    lambda_t = np.where(rate_change > 0, lambda_up, lambda_down)
    return beta_level * (1 - lambda_t * vol_ratio)

# Calculate model predictions
Y = df['ILMDHYLD'].values
FEDL01 = df['FEDL01'].values
FHLK3MSPRD = df['FHLK3MSPRD'].values
vol_ratio = df['vol_ratio'].values
rate_change = df['FEDL01_change'].fillna(0).values

# Dynamic beta
beta_dynamic = asymmetric_volatility_beta(
    FEDL01, PARAMS['k'], PARAMS['m'], PARAMS['beta_min'], PARAMS['beta_max'],
    vol_ratio, PARAMS['lambda_up'], PARAMS['lambda_down'], rate_change
)

# Predictions
pred_dynamic = PARAMS['alpha'] + beta_dynamic * FEDL01 + PARAMS['gamma_fhlb'] * FHLK3MSPRD
pred_static = STATIC_ALPHA + STATIC_BETA * FEDL01

# Calculate metrics
rmse_dynamic = np.sqrt(np.mean((Y - pred_dynamic)**2))
rmse_static = np.sqrt(np.mean((Y - pred_static)**2))
r2_dynamic = 1 - np.sum((Y - pred_dynamic)**2) / np.sum((Y - np.mean(Y))**2)
r2_static = 1 - np.sum((Y - pred_static)**2) / np.sum((Y - np.mean(Y))**2)

print(f"\nModel Performance:")
print(f"  Dynamic Model: R² = {r2_dynamic:.4f}, RMSE = {rmse_dynamic:.4f}%")
print(f"  Static OLS:    R² = {r2_static:.4f}, RMSE = {rmse_static:.4f}%")

# =============================================================================
# FIGURE 1: Data Dashboard
# =============================================================================
print("\nGenerating Figure 1: Data Dashboard...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel A: Time series
ax1 = axes[0, 0]
ax1.plot(df['EOM_Dt'], df['ILMDHYLD'], 'b-', linewidth=1.5, label='MMDA Rate')
ax1.plot(df['EOM_Dt'], df['FEDL01'], 'r--', linewidth=1.5, label='Fed Funds Rate')
ax1.set_xlabel('Date')
ax1.set_ylabel('Rate (%)')
ax1.set_title('(a) Historical Rate Evolution')
ax1.legend(loc='upper left')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax1.axvspan(pd.Timestamp('2022-03-01'), pd.Timestamp('2025-03-31'), 
            alpha=0.15, color='gold', label='Hiking Cycle')

# Panel B: Scatter plot
ax2 = axes[0, 1]
ax2.scatter(df['FEDL01'], df['ILMDHYLD'], c=df['vol_ratio'], cmap='RdYlBu_r', 
            alpha=0.7, s=40, edgecolors='k', linewidth=0.3)
ax2.plot([0, 6], [STATIC_ALPHA, STATIC_ALPHA + STATIC_BETA * 6], 'r--', 
         linewidth=2, label=f'Static β = {STATIC_BETA:.1%}')
ax2.set_xlabel('Fed Funds Rate (%)')
ax2.set_ylabel('MMDA Rate (%)')
ax2.set_title('(b) MMDA vs Fed Funds Relationship')
ax2.legend(loc='upper left')
cbar = plt.colorbar(ax2.collections[0], ax=ax2)
cbar.set_label('Volatility Ratio')

# Panel C: Rolling 24-month OLS beta
from numpy.lib.stride_tricks import sliding_window_view
window = 24
rolling_ols_beta = pd.Series(index=df.index, dtype=float)
for i in range(window, len(df) + 1):
    y = df['ILMDHYLD'].iloc[i-window:i].values
    x = df['FEDL01'].iloc[i-window:i].values
    x_mean = x.mean()
    y_mean = y.mean()
    beta = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean)**2) if np.sum((x - x_mean)**2) > 1e-10 else np.nan
    rolling_ols_beta.iloc[i-1] = beta
ax3 = axes[1, 0]
ax3.plot(df['EOM_Dt'], rolling_ols_beta, 'g-', linewidth=1.5, label='24-month rolling OLS β')
ax3.axhline(y=STATIC_BETA, color='r', linestyle='--', label=f'Static β = {STATIC_BETA:.1%}')
ax3.axhline(y=PARAMS['beta_min'], color='b', linestyle=':', alpha=0.7, label=f'β_min = {PARAMS["beta_min"]:.0%}')
ax3.axhline(y=PARAMS['beta_max'], color='b', linestyle=':', alpha=0.7, label=f'β_max = {PARAMS["beta_max"]:.0%}')
ax3.set_xlabel('Date')
ax3.set_ylabel('Rolling Beta (24-month OLS)')
ax3.set_title('(c) Time-Varying Deposit Beta')
ax3.legend(loc='upper left', fontsize=9)
ax3.set_ylim(0.0, 1.0)
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# Panel D: Volatility
ax4 = axes[1, 1]
ax4.fill_between(df['EOM_Dt'], 0, df['rolling_vol'], alpha=0.4, color='purple')
ax4.plot(df['EOM_Dt'], df['rolling_vol'], 'purple', linewidth=1.5)
ax4.axhline(y=df['rolling_vol'].mean(), color='k', linestyle='--', 
            label=f'Mean = {df["rolling_vol"].mean():.2f}%')
ax4.set_xlabel('Date')
ax4.set_ylabel('24-Month Rolling Volatility (%)')
ax4.set_title('(d) Fed Funds Rate Volatility')
ax4.legend(loc='upper right')
ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
plt.savefig('outputs/visualizations/01_data_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: outputs/visualizations/01_data_dashboard.png")

# =============================================================================
# FIGURE 2: Model Fit Comparison
# =============================================================================
print("\nGenerating Figure 2: Model Fit Comparison...")

fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Panel A: Full time series comparison
ax1 = axes[0]
ax1.plot(df['EOM_Dt'], Y, 'ko-', markersize=4, linewidth=1, label='Actual MMDA', alpha=0.8)
ax1.plot(df['EOM_Dt'], pred_dynamic, 'b-', linewidth=2, label=f'Dynamic Model (RMSE={rmse_dynamic:.3f}%)')
ax1.plot(df['EOM_Dt'], pred_static, 'r--', linewidth=2, label=f'Static OLS β=46.2% (RMSE={rmse_static:.3f}%)')
ax1.axvspan(pd.Timestamp('2022-03-01'), pd.Timestamp('2025-03-31'), 
            alpha=0.15, color='gold', label='2022-25 Hiking Cycle')
ax1.set_xlabel('Date')
ax1.set_ylabel('MMDA Rate (%)')
ax1.set_title('Model Fit Comparison: Dynamic vs Static Beta')
ax1.legend(loc='upper left')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# Panel B: Residuals comparison
ax2 = axes[1]
resid_dynamic = Y - pred_dynamic
resid_static = Y - pred_static
ax2.bar(df['EOM_Dt'], resid_static, width=25, alpha=0.5, color='red', label=f'Static OLS Residuals')
ax2.plot(df['EOM_Dt'], resid_dynamic, 'b-', linewidth=2, label=f'Dynamic Model Residuals')
ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
ax2.axvspan(pd.Timestamp('2022-03-01'), pd.Timestamp('2025-03-31'), 
            alpha=0.15, color='gold')
ax2.set_xlabel('Date')
ax2.set_ylabel('Residual (%)')
ax2.set_title('Residual Comparison: Static Model Shows Systematic Errors During Rate Transitions')
ax2.legend(loc='upper left')
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
plt.savefig('outputs/visualizations/02_model_fit_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: outputs/visualizations/02_model_fit_comparison.png")

# =============================================================================
# FIGURE 3: Asymmetric Beta Evolution
# =============================================================================
print("\nGenerating Figure 3: Asymmetric Beta Evolution...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

fed_range = np.linspace(0, 6, 100)
vol_levels = [0.5, 1.0, 1.5, 2.0]
colors = ['#2166ac', '#67a9cf', '#ef8a62', '#b2182b']

# Panel A: Rising rates (lambda_up)
ax1 = axes[0]
for vol, color in zip(vol_levels, colors):
    beta = logistic_beta(fed_range, PARAMS['k'], PARAMS['m'], PARAMS['beta_min'], PARAMS['beta_max'])
    beta_adj = beta * (1 - PARAMS['lambda_up'] * vol)
    ax1.plot(fed_range, beta_adj * 100, color=color, linewidth=2, 
             label=f'Vol Ratio = {vol:.1f}')

ax1.axvline(x=PARAMS['m'], color='gray', linestyle='--', alpha=0.7)
ax1.text(PARAMS['m'] + 0.1, 35, f'm = {PARAMS["m"]:.2f}%', fontsize=10, color='gray')
ax1.axhline(y=STATIC_BETA * 100, color='red', linestyle=':', linewidth=2, 
            label=f'Static β = {STATIC_BETA:.1%}')
ax1.set_xlabel('Fed Funds Rate (%)')
ax1.set_ylabel('Deposit Beta (%)')
ax1.set_title(f'(a) Rising Rates: λ_up = {PARAMS["lambda_up"]:.1%} Dampening')
ax1.legend(loc='upper left', fontsize=9)
ax1.set_xlim(0, 6)
ax1.set_ylim(20, 75)
ax1.grid(True, alpha=0.3)

# Panel B: Falling rates (lambda_down)
ax2 = axes[1]
for vol, color in zip(vol_levels, colors):
    beta = logistic_beta(fed_range, PARAMS['k'], PARAMS['m'], PARAMS['beta_min'], PARAMS['beta_max'])
    beta_adj = beta * (1 - PARAMS['lambda_down'] * vol)
    ax2.plot(fed_range, beta_adj * 100, color=color, linewidth=2, 
             label=f'Vol Ratio = {vol:.1f}')

ax2.axvline(x=PARAMS['m'], color='gray', linestyle='--', alpha=0.7)
ax2.text(PARAMS['m'] + 0.1, 35, f'm = {PARAMS["m"]:.2f}%', fontsize=10, color='gray')
ax2.axhline(y=STATIC_BETA * 100, color='red', linestyle=':', linewidth=2, 
            label=f'Static β = {STATIC_BETA:.1%}')
ax2.set_xlabel('Fed Funds Rate (%)')
ax2.set_ylabel('Deposit Beta (%)')
ax2.set_title(f'(b) Falling Rates: λ_down = {PARAMS["lambda_down"]:.1%} Dampening')
ax2.legend(loc='upper left', fontsize=9)
ax2.set_xlim(0, 6)
ax2.set_ylim(20, 75)
ax2.grid(True, alpha=0.3)

plt.suptitle(f'Asymmetric Volatility Dampening: λ_up - λ_down = {(PARAMS["lambda_up"]-PARAMS["lambda_down"])*100:.1f} pp', 
             fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('outputs/figures/figure3_asymmetric_beta_evolution.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: outputs/figures/figure3_asymmetric_beta_evolution.png")

# =============================================================================
# FIGURE 4: Residual Analysis
# =============================================================================
print("\nGenerating Figure 4: Residual Analysis...")

residuals = Y - pred_dynamic

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel A: Residual time series
ax1 = axes[0, 0]
ax1.plot(df['EOM_Dt'], residuals, 'b-', linewidth=1)
ax1.axhline(y=0, color='r', linestyle='--')
ax1.fill_between(df['EOM_Dt'], residuals, 0, alpha=0.3)
ax1.set_xlabel('Date')
ax1.set_ylabel('Residual (%)')
ax1.set_title('(a) Residual Time Series')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# Panel B: Histogram
ax2 = axes[0, 1]
ax2.hist(residuals, bins=20, density=True, alpha=0.7, color='steelblue', edgecolor='black')
from scipy import stats
x_norm = np.linspace(residuals.min(), residuals.max(), 100)
ax2.plot(x_norm, stats.norm.pdf(x_norm, residuals.mean(), residuals.std()), 
         'r-', linewidth=2, label='Normal Distribution')
ax2.set_xlabel('Residual (%)')
ax2.set_ylabel('Density')
ax2.set_title('(b) Residual Distribution')
ax2.legend()

# Panel C: Q-Q plot
ax3 = axes[1, 0]
stats.probplot(residuals, dist="norm", plot=ax3)
ax3.set_title('(c) Q-Q Plot')
ax3.get_lines()[0].set_markerfacecolor('steelblue')
ax3.get_lines()[0].set_markeredgecolor('black')

# Panel D: Residuals vs Fitted
ax4 = axes[1, 1]
ax4.scatter(pred_dynamic, residuals, alpha=0.6, c='steelblue', edgecolors='black', linewidth=0.3)
ax4.axhline(y=0, color='r', linestyle='--')
ax4.set_xlabel('Fitted Values (%)')
ax4.set_ylabel('Residual (%)')
ax4.set_title('(d) Residuals vs Fitted Values')

# Add RMSE annotation
textstr = f'RMSE = {rmse_dynamic:.4f}%\nR² = {r2_dynamic:.4f}'
ax4.text(0.95, 0.95, textstr, transform=ax4.transAxes, fontsize=10,
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('outputs/visualizations/04_residual_analysis_vol_adjusted.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: outputs/visualizations/04_residual_analysis_vol_adjusted.png")

# =============================================================================
# FIGURE 5: Beta Evolution Over Time
# =============================================================================
print("\nGenerating Figure 5: Beta Evolution...")

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(df['EOM_Dt'], beta_dynamic * 100, 'b-', linewidth=2, label='Dynamic Beta (Asymmetric Vol-Adj)')
ax.axhline(y=STATIC_BETA * 100, color='red', linestyle='--', linewidth=2, label=f'Static OLS β = {STATIC_BETA:.1%}')
ax.axhline(y=PARAMS['beta_min'] * 100, color='gray', linestyle=':', alpha=0.7, label=f'β_min = {PARAMS["beta_min"]:.0%}')
ax.axhline(y=PARAMS['beta_max'] * 100, color='gray', linestyle=':', alpha=0.7, label=f'β_max = {PARAMS["beta_max"]:.0%}')
ax.axvspan(pd.Timestamp('2022-03-01'), pd.Timestamp('2025-03-31'), 
           alpha=0.15, color='gold', label='Hiking Cycle')

ax.set_xlabel('Date')
ax.set_ylabel('Deposit Beta (%)')
ax.set_title('Evolution of Asymmetric Volatility-Adjusted Deposit Beta')
ax.legend(loc='upper left')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.set_ylim(20, 80)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/visualizations/03_beta_evolution.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: outputs/visualizations/03_beta_evolution.png")

print("\n" + "="*60)
print("All figures regenerated successfully!")
print("="*60)
