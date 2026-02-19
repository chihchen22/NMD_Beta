"""
Regenerate figures with Static OLS benchmark included
Updates Figure 2 (Model Fit Comparison) and Beta Evolution Chart
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm
from mmda_dynamic_beta_model_complete import MMDADynamicBetaModel
import os

print("=" * 80)
print("REGENERATING FIGURES WITH STATIC OLS BENCHMARK")
print("=" * 80)

# Initialize and load data
model = MMDADynamicBetaModel()
model.load_and_prepare_data('bankratemma.csv')

# Run full analysis to get all model results
print("\n1. Running full model analysis...")
model.run_full_analysis()

# Extract variables
Y = model.data['ILMDHYLD'].values
FEDL01 = model.data['FEDL01'].values
FHLK3MSPRD = model.data['FHLK3MSPRD'].values
term_spread = model.data['1Y_3M_SPRD'].values
vol_ratio = model.data['vol_ratio'].values
rate_change = model.data['FEDL01_change'].fillna(0).values
dates = model.data['EOM_Dt']
n = len(Y)

# ============================================================
# Estimate Static OLS
# ============================================================
print("\n2. Estimating Static OLS benchmark...")
X_simple = sm.add_constant(FEDL01)
ols_simple = sm.OLS(Y, X_simple).fit()
ols_fitted = ols_simple.fittedvalues
ols_beta = ols_simple.params[1]
ols_alpha = ols_simple.params[0]
ols_r2 = ols_simple.rsquared
ols_rmse = np.sqrt(np.mean((Y - ols_fitted)**2))

# Recent period RMSE
recent_mask = model.data['EOM_Dt'] >= '2022-03-01'
ols_recent_rmse = np.sqrt(np.mean((Y[recent_mask] - ols_fitted[recent_mask])**2))

print(f"   Static Beta: {ols_beta:.4f}")
print(f"   R²: {ols_r2:.4f}")
print(f"   RMSE: {ols_rmse:.4f}%")
print(f"   2022-25 RMSE: {ols_recent_rmse:.4f}%")

# ============================================================
# Estimate Asymmetric Model (our primary model)
# ============================================================
print("\n3. Estimating Asymmetric volatility model...")
asym_result = model.estimate_asymmetric_volatility(Y, FEDL01, FHLK3MSPRD, term_spread, vol_ratio, rate_change)
asym_params = asym_result.x
alpha, k, m, beta_min, beta_max, gamma_fhlb, gamma_term, lambda_up, lambda_down = asym_params

# Calculate asymmetric model fitted values
dynamic_beta = model.asymmetric_volatility_beta(FEDL01, k, m, beta_min, beta_max, 
                                                 vol_ratio, lambda_up, lambda_down, rate_change)
asym_fitted = alpha + dynamic_beta * FEDL01 + gamma_fhlb * FHLK3MSPRD + gamma_term * term_spread
asym_r2 = 1 - np.sum((Y - asym_fitted)**2) / np.sum((Y - np.mean(Y))**2)
asym_rmse = np.sqrt(np.mean((Y - asym_fitted)**2))
asym_recent_rmse = np.sqrt(np.mean((Y[recent_mask] - asym_fitted[recent_mask])**2))

print(f"   R²: {asym_r2:.4f}")
print(f"   RMSE: {asym_rmse:.4f}%")
print(f"   2022-25 RMSE: {asym_recent_rmse:.4f}%")

# ============================================================
# Get other model results from existing analysis
# ============================================================
enhanced_fitted = model.results['models']['enhanced']['predictions']
enhanced_r2 = model.results['models']['enhanced']['metrics']['r_squared']
enhanced_rmse = model.results['models']['enhanced']['metrics']['rmse']
enhanced_recent_rmse = model.results['models']['enhanced']['recent_rmse']

quadratic_fitted = model.results['models']['quadratic']['predictions']
quadratic_r2 = model.results['models']['quadratic']['metrics']['r_squared']
quadratic_rmse = model.results['models']['quadratic']['metrics']['rmse']
quadratic_recent_rmse = model.results['models']['quadratic']['recent_rmse']

# ============================================================
# FIGURE 2: Model Fit Comparison (with Static OLS)
# ============================================================
print("\n4. Creating Figure 2: Model Fit Comparison...")

fig, ax = plt.subplots(figsize=(16, 10))

# Plot actual data
ax.plot(dates, Y, 'ko', markersize=6, markeredgecolor='white',
        markeredgewidth=1, label='Actual MMDA Rate', alpha=0.8, zorder=5)

# Plot Static OLS (new - dashed orange)
ax.plot(dates, ols_fitted, color='#FF8C00', linestyle='--', linewidth=2.5, alpha=0.8,
        label=f"Static OLS β={ols_beta:.1%} (R²={ols_r2:.3f})")

# Plot Enhanced Logistic (red)
ax.plot(dates, enhanced_fitted, color='#E74C3C', linestyle='-', linewidth=2.0, alpha=0.8,
        label=f"Enhanced Logistic (R²={enhanced_r2:.3f})")

# Plot Quadratic (green dashed)
ax.plot(dates, quadratic_fitted, color='#2ECC71', linestyle='--', linewidth=2.0, alpha=0.8,
        label=f"Quadratic (R²={quadratic_r2:.3f})")

# Plot Asymmetric Vol-Adjusted (blue - primary)
ax.plot(dates, asym_fitted, color='#3498DB', linestyle='-', linewidth=3.0, alpha=0.9,
        label=f"Asymmetric Vol-Adjusted (R²={asym_r2:.3f})")

# Highlight 2022-2025 period
recent_start = dates[recent_mask].iloc[0]
recent_end = dates.iloc[-1]
ax.axvspan(recent_start, recent_end, alpha=0.15, color='gold',
          label='2022-2025 Focus Period', zorder=1)

# Formatting
ax.set_xlabel('Date', fontsize=14, fontweight='bold')
ax.set_ylabel('MMDA Rate (%)', fontsize=14, fontweight='bold')
ax.set_title('MMDA Model Fit Comparison: Static vs Dynamic Beta Approaches (2017-2025)',
            fontsize=16, fontweight='bold', pad=20)
ax.legend(fontsize=11, loc='upper left', framealpha=0.9)
ax.grid(True, alpha=0.3)

# Add performance statistics box
textstr = "2022-2025 RMSE Comparison:\n"
textstr += f"• Static OLS (β={ols_beta:.1%}): {ols_recent_rmse:.3f}%\n"
textstr += f"• Enhanced Logistic: {enhanced_recent_rmse:.3f}%\n"
textstr += f"• Quadratic: {quadratic_recent_rmse:.3f}%\n"
textstr += f"• Asymmetric Vol-Adj: {asym_recent_rmse:.3f}%\n"
textstr += f"\nDynamic vs Static: {(ols_recent_rmse - asym_recent_rmse)/ols_recent_rmse*100:.0f}% improvement"

props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray')
ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=10,
       verticalalignment='bottom', horizontalalignment='right', 
       bbox=props, family='monospace')

plt.tight_layout()
plt.savefig('outputs/visualizations/02_model_fit_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('outputs/visualizations/02_model_fit_comparison.pdf', dpi=300, bbox_inches='tight', facecolor='white')
print("   Saved: outputs/visualizations/02_model_fit_comparison.png")
plt.close()

# ============================================================
# FIGURE: Beta Evolution Comparison (with Static OLS)
# ============================================================
print("\n5. Creating Beta Evolution Chart...")

fig, ax = plt.subplots(figsize=(14, 9))

# Create rate grid
rate_grid = np.linspace(0, 6, 100)
vol_ratio_mean = np.mean(vol_ratio)

# Static OLS - constant beta (horizontal line)
static_beta_grid = np.full_like(rate_grid, ols_beta)
ax.plot(rate_grid, static_beta_grid, color='#FF8C00', linestyle='--', linewidth=3,
        label=f'Static OLS (β = {ols_beta:.1%})', alpha=0.9)

# Enhanced Logistic
enhanced_params = model.results['models']['enhanced']['params']
enhanced_beta_grid = model.logistic_beta(rate_grid, enhanced_params['k'], enhanced_params['m'],
                                         enhanced_params['beta_min'], enhanced_params['beta_max'])
ax.plot(rate_grid, enhanced_beta_grid, color='#E74C3C', linestyle='-', linewidth=2.5,
        marker='o', markersize=6, markevery=15, markeredgecolor='white',
        label='Enhanced Logistic', alpha=0.8)

# Quadratic
quad_params = model.results['models']['quadratic']['params']
quadratic_beta_grid = model.quadratic_beta(rate_grid, quad_params['a'], quad_params['b'], 
                                           quad_params['c'], quad_params['beta_min'], quad_params['beta_max'])
ax.plot(rate_grid, quadratic_beta_grid, color='#2ECC71', linestyle='--', linewidth=2.5,
        marker='^', markersize=6, markevery=15, markeredgecolor='white',
        label='Quadratic', alpha=0.8)

# Asymmetric Vol-Adjusted (average volatility, rising rates)
asym_beta_rising = model.asymmetric_volatility_beta(rate_grid, k, m, beta_min, beta_max,
                                                     vol_ratio_mean, lambda_up, lambda_down,
                                                     np.ones_like(rate_grid) * 0.1)  # positive = rising
ax.plot(rate_grid, asym_beta_rising, color='#3498DB', linestyle='-', linewidth=3,
        marker='s', markersize=6, markevery=15, markeredgecolor='white',
        label='Asymmetric Vol-Adj (Rising)', alpha=0.9)

# Asymmetric Vol-Adjusted (average volatility, falling rates)
asym_beta_falling = model.asymmetric_volatility_beta(rate_grid, k, m, beta_min, beta_max,
                                                      vol_ratio_mean, lambda_up, lambda_down,
                                                      np.ones_like(rate_grid) * -0.1)  # negative = falling
ax.plot(rate_grid, asym_beta_falling, color='#9B59B6', linestyle='-', linewidth=2.5,
        marker='D', markersize=5, markevery=15, markeredgecolor='white',
        label='Asymmetric Vol-Adj (Falling)', alpha=0.8)

# Mark inflection point
inflection_beta = model.logistic_beta(np.array([m]), k, m, beta_min, beta_max)[0]
ax.scatter([m], [inflection_beta], color='red', s=150, marker='*', edgecolor='white', 
           linewidth=2, zorder=6, label=f'Inflection Point (m={m:.1f}%)')

# Reference lines
for level in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
    ax.axhline(y=level, color='gray', linestyle=':', alpha=0.3, linewidth=1)

# Formatting
ax.set_xlabel('Federal Funds Rate (%)', fontsize=14, fontweight='bold')
ax.set_ylabel('Deposit Beta (Rate Sensitivity)', fontsize=14, fontweight='bold')
ax.set_title('Deposit Beta Evolution: Static vs Dynamic Approaches',
            fontsize=16, fontweight='bold', pad=20)
ax.set_xlim(0, 6)
ax.set_ylim(0.30, 0.75)
ax.legend(fontsize=10, loc='lower right', framealpha=0.9)
ax.grid(True, alpha=0.3)

# Add annotation
ax.annotate('Static beta misses\nrate-level dependence', 
            xy=(4.5, ols_beta), xytext=(4.5, 0.55),
            fontsize=10, ha='center',
            arrowprops=dict(arrowstyle='->', color='#FF8C00', lw=1.5))

plt.tight_layout()
plt.savefig('outputs/visualizations/03_beta_evolution_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('outputs/visualizations/03_beta_evolution_comparison.pdf', dpi=300, bbox_inches='tight', facecolor='white')
print("   Saved: outputs/visualizations/03_beta_evolution_comparison.png")
plt.close()

# ============================================================
# FIGURE: Residual Comparison (Static vs Dynamic)
# ============================================================
print("\n6. Creating Residual Comparison Chart...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Panel A: Residuals over time
ax1 = axes[0, 0]
ax1.plot(dates, Y - ols_fitted, color='#FF8C00', alpha=0.7, linewidth=1.5, label='Static OLS')
ax1.plot(dates, Y - asym_fitted, color='#3498DB', alpha=0.7, linewidth=1.5, label='Asymmetric Dynamic')
ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax1.axvspan(recent_start, recent_end, alpha=0.15, color='gold')
ax1.set_xlabel('Date', fontsize=12)
ax1.set_ylabel('Residual (%)', fontsize=12)
ax1.set_title('(A) Residuals Over Time', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Panel B: Residuals vs Fed Funds Rate
ax2 = axes[0, 1]
ax2.scatter(FEDL01, Y - ols_fitted, color='#FF8C00', alpha=0.5, s=40, label='Static OLS')
ax2.scatter(FEDL01, Y - asym_fitted, color='#3498DB', alpha=0.5, s=40, label='Asymmetric Dynamic')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax2.set_xlabel('Federal Funds Rate (%)', fontsize=12)
ax2.set_ylabel('Residual (%)', fontsize=12)
ax2.set_title('(B) Residuals vs Rate Level', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# Panel C: Cumulative squared residuals
ax3 = axes[1, 0]
cum_sq_ols = np.cumsum((Y - ols_fitted)**2)
cum_sq_asym = np.cumsum((Y - asym_fitted)**2)
ax3.plot(dates, cum_sq_ols, color='#FF8C00', linewidth=2, label='Static OLS')
ax3.plot(dates, cum_sq_asym, color='#3498DB', linewidth=2, label='Asymmetric Dynamic')
ax3.axvspan(recent_start, recent_end, alpha=0.15, color='gold')
ax3.set_xlabel('Date', fontsize=12)
ax3.set_ylabel('Cumulative Squared Error', fontsize=12)
ax3.set_title('(C) Cumulative Squared Error', fontsize=13, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# Panel D: RMSE by year
ax4 = axes[1, 1]
years = range(2017, 2026)
ols_rmse_by_year = []
asym_rmse_by_year = []
for year in years:
    year_mask = model.data['EOM_Dt'].dt.year == year
    if year_mask.sum() > 0:
        ols_rmse_by_year.append(np.sqrt(np.mean((Y[year_mask] - ols_fitted[year_mask])**2)))
        asym_rmse_by_year.append(np.sqrt(np.mean((Y[year_mask] - asym_fitted[year_mask])**2)))
    else:
        ols_rmse_by_year.append(0)
        asym_rmse_by_year.append(0)

x = np.arange(len(years))
width = 0.35
ax4.bar(x - width/2, ols_rmse_by_year, width, label='Static OLS', color='#FF8C00', alpha=0.8)
ax4.bar(x + width/2, asym_rmse_by_year, width, label='Asymmetric Dynamic', color='#3498DB', alpha=0.8)
ax4.set_xlabel('Year', fontsize=12)
ax4.set_ylabel('RMSE (%)', fontsize=12)
ax4.set_title('(D) RMSE by Year', fontsize=13, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(years)
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3, axis='y')

plt.suptitle('Static vs Dynamic Model: Residual Analysis', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('outputs/visualizations/05_residual_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
print("   Saved: outputs/visualizations/05_residual_comparison.png")
plt.close()

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 80)
print("FIGURES REGENERATED SUCCESSFULLY")
print("=" * 80)
print("""
Updated figures:
1. outputs/visualizations/02_model_fit_comparison.png - Now includes Static OLS
2. outputs/visualizations/03_beta_evolution_comparison.png - Shows constant vs dynamic beta
3. outputs/visualizations/05_residual_comparison.png - Residual analysis comparison

Key visual insights:
- Static OLS (orange dashed) shows systematic deviations during rate transitions
- Dynamic models (blue solid) capture the rate-level dependence
- 2022-2025 period shows largest gap between static and dynamic approaches
- Beta evolution chart clearly shows static β=46.2% vs dynamic range of 40-70%
""")
