"""
Run asymmetric volatility analysis as primary specification (Critique #1)
Generate Figure 3 with separate panels for rising vs falling rate environments
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mmda_dynamic_beta_model_complete import MMDADynamicBetaModel

print("=" * 80)
print("CRITIQUE #1: ASYMMETRIC VOLATILITY AS PRIMARY SPECIFICATION")
print("=" * 80)

# Initialize and load data
model = MMDADynamicBetaModel()
model.load_and_prepare_data('bankratemma.csv')

# Extract variables
Y = model.data['ILMDHYLD'].values
FEDL01 = model.data['FEDL01'].values
FHLK3MSPRD = model.data['FHLK3MSPRD'].values
term_spread = model.data['1Y_3M_SPRD'].values
vol_ratio = model.data['vol_ratio'].values
rate_change = model.data['FEDL01_change'].fillna(0).values

print("\n1. Estimating Asymmetric Volatility Model (Primary Specification)...")
print("-" * 60)

# Estimate asymmetric model
asym_result = model.estimate_asymmetric_volatility(Y, FEDL01, FHLK3MSPRD, term_spread, vol_ratio, rate_change)

if asym_result.success:
    params = asym_result.x
    print("   Optimization successful!")
    print("\n   ASYMMETRIC MODEL PARAMETERS:")
    print(f"   alpha      = {params[0]:.4f}  (Base margin)")
    print(f"   k          = {params[1]:.4f}  (Transition speed)")
    print(f"   m          = {params[2]:.4f}  (Inflection point)")
    print(f"   beta_min   = {params[3]:.4f}  (Minimum sensitivity)")
    print(f"   beta_max   = {params[4]:.4f}  (Maximum sensitivity)")
    print(f"   gamma_fhlb = {params[5]:.4f}  (FHLB spread coefficient)")
    print(f"   gamma_term = {params[6]:.4f}  (Term spread coefficient)")
    print(f"   lambda_up  = {params[7]:.4f}  (Rising rate dampening)")
    print(f"   lambda_down= {params[8]:.4f}  (Falling rate dampening)")
    
    # Calculate predictions
    beta_asym = model.asymmetric_volatility_beta(
        FEDL01, params[1], params[2], params[3], params[4],
        vol_ratio, params[7], params[8], rate_change
    )
    pred_asym = params[0] + beta_asym * FEDL01 + params[5] * FHLK3MSPRD + params[6] * term_spread
    
    # Metrics
    n = len(Y)
    rmse = np.sqrt(np.mean((Y - pred_asym)**2))
    r2 = 1 - np.sum((Y - pred_asym)**2) / np.sum((Y - Y.mean())**2)
    ll = -asym_result.fun
    aic = -2 * ll + 2 * 9
    bic = -2 * ll + np.log(n) * 9
    
    print("\n   MODEL FIT:")
    print(f"   R-squared  = {r2:.4f}")
    print(f"   RMSE       = {rmse:.4f}%")
    print(f"   AIC        = {aic:.1f}")
    print(f"   BIC        = {bic:.1f}")
    
    # Store asymmetric results
    asym_params = {
        'alpha': params[0],
        'k': params[1],
        'm': params[2],
        'beta_min': params[3],
        'beta_max': params[4],
        'gamma_fhlb': params[5],
        'gamma_term': params[6],
        'lambda_up': params[7],
        'lambda_down': params[8]
    }
    asym_metrics = {
        'r_squared': r2,
        'rmse': rmse,
        'aic': aic,
        'bic': bic,
        'log_likelihood': ll
    }

# Compare with symmetric baseline
print("\n2. Comparing with Symmetric Baseline...")
print("-" * 60)

sym_result = model.estimate_volatility_adjusted(Y, FEDL01, FHLK3MSPRD, term_spread, vol_ratio)
if sym_result.success:
    sym_params = sym_result.x
    beta_sym = model.volatility_adjusted_beta(
        FEDL01, sym_params[1], sym_params[2], sym_params[3], sym_params[4],
        vol_ratio, sym_params[7]
    )
    pred_sym = sym_params[0] + beta_sym * FEDL01 + sym_params[5] * FHLK3MSPRD + sym_params[6] * term_spread
    
    rmse_sym = np.sqrt(np.mean((Y - pred_sym)**2))
    r2_sym = 1 - np.sum((Y - pred_sym)**2) / np.sum((Y - Y.mean())**2)
    ll_sym = -sym_result.fun
    aic_sym = -2 * ll_sym + 2 * 8
    
    print(f"   Symmetric lambda = {sym_params[7]:.4f}")
    print(f"   Symmetric RMSE   = {rmse_sym:.4f}%")
    print(f"   Symmetric R2     = {r2_sym:.4f}")
    
    # Likelihood ratio test
    lr_stat = 2 * (ll - ll_sym)
    from scipy.stats import chi2
    p_value = 1 - chi2.cdf(lr_stat, df=1)
    
    print("\n   LIKELIHOOD RATIO TEST (Symmetric vs Asymmetric):")
    print(f"   LR Statistic = {lr_stat:.2f}")
    print(f"   P-value      = {p_value:.6f}")
    if p_value < 0.05:
        print("   Result: REJECT symmetry - Asymmetric model preferred")
    else:
        print("   Result: Fail to reject symmetry")
    
    improvement = (rmse_sym - rmse) / rmse_sym * 100
    print(f"\n   RMSE Improvement: {improvement:.1f}%")

# Compare with other models
print("\n3. Model Comparison Table...")
print("-" * 60)

# Enhanced logistic
enh_result = model.estimate_enhanced_logistic(Y, FEDL01, FHLK3MSPRD, term_spread)
if enh_result.success:
    enh_params = enh_result.x
    beta_enh = model.logistic_beta(FEDL01, enh_params[1], enh_params[2], enh_params[3], enh_params[4])
    pred_enh = enh_params[0] + beta_enh * FEDL01 + enh_params[5] * FHLK3MSPRD + enh_params[6] * term_spread
    rmse_enh = np.sqrt(np.mean((Y - pred_enh)**2))
    r2_enh = 1 - np.sum((Y - pred_enh)**2) / np.sum((Y - Y.mean())**2)
    ll_enh = -enh_result.fun
    aic_enh = -2 * ll_enh + 2 * 7

# Quadratic
quad_result = model.estimate_quadratic(Y, FEDL01, FHLK3MSPRD, term_spread)
if quad_result.success:
    quad_params = quad_result.x
    beta_quad = model.quadratic_beta(FEDL01, quad_params[1], quad_params[2], quad_params[3], quad_params[4], quad_params[5])
    pred_quad = quad_params[0] + beta_quad * FEDL01 + quad_params[6] * FHLK3MSPRD + quad_params[7] * term_spread
    rmse_quad = np.sqrt(np.mean((Y - pred_quad)**2))
    r2_quad = 1 - np.sum((Y - pred_quad)**2) / np.sum((Y - Y.mean())**2)
    ll_quad = -quad_result.fun
    aic_quad = -2 * ll_quad + 2 * 8

print("\n   Model                      R2       RMSE        AIC    Params")
print("   " + "-" * 65)
print(f"   Asymmetric Vol-Adj*     {r2:.4f}   {rmse:.4f}%   {aic:>8.1f}        9")
print(f"   Symmetric Vol-Adj       {r2_sym:.4f}   {rmse_sym:.4f}%   {aic_sym:>8.1f}        8")
print(f"   Enhanced Logistic       {r2_enh:.4f}   {rmse_enh:.4f}%   {aic_enh:>8.1f}        7")
print(f"   Quadratic               {r2_quad:.4f}   {rmse_quad:.4f}%   {aic_quad:>8.1f}        8")
print("   * Recommended primary specification")

# Recent period comparison (2022-2025)
print("\n4. Recent Period Performance (2022-2025)...")
print("-" * 60)

recent_mask = model.data['EOM_Dt'] >= '2022-01-01'
Y_recent = Y[recent_mask]

rmse_recent_asym = np.sqrt(np.mean((Y_recent - pred_asym[recent_mask])**2))
rmse_recent_sym = np.sqrt(np.mean((Y_recent - pred_sym[recent_mask])**2))
rmse_recent_enh = np.sqrt(np.mean((Y_recent - pred_enh[recent_mask])**2))
rmse_recent_quad = np.sqrt(np.mean((Y_recent - pred_quad[recent_mask])**2))

print("\n   Model                    Recent RMSE    Improvement vs Asym")
print("   " + "-" * 60)
print(f"   Asymmetric Vol-Adj*      {rmse_recent_asym:.4f}%        --")
imp_sym = (rmse_recent_sym - rmse_recent_asym)/rmse_recent_sym*100
imp_enh = (rmse_recent_enh - rmse_recent_asym)/rmse_recent_enh*100
imp_quad = (rmse_recent_quad - rmse_recent_asym)/rmse_recent_quad*100
print(f"   Symmetric Vol-Adj        {rmse_recent_sym:.4f}%        {imp_sym:+.1f}%")
print(f"   Enhanced Logistic        {rmse_recent_enh:.4f}%        {imp_enh:+.1f}%")
print(f"   Quadratic                {rmse_recent_quad:.4f}%        {imp_quad:+.1f}%")

# Generate Figure 3: Asymmetric Beta Evolution with Separate Panels
print("\n5. Generating Figure 3: Asymmetric Beta Evolution...")
print("-" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Create Fed Funds rate range for plotting
ff_range = np.linspace(0, 6, 100)

# Panel A: Rising Rate Environment
ax1 = axes[0]
ax1.set_title('Panel A: Rising Rate Environment', fontsize=12, fontweight='bold')

# Different volatility levels
for vol_level, vol_label, color in [(0.5, 'Low Vol (0.5x)', '#2ecc71'), 
                                     (1.0, 'Normal Vol (1.0x)', '#3498db'),
                                     (2.0, 'High Vol (2.0x)', '#e74c3c')]:
    # Rising rates: use lambda_up
    beta_rising = model.logistic_beta(ff_range, asym_params['k'], asym_params['m'], 
                                       asym_params['beta_min'], asym_params['beta_max'])
    vol_adj_rising = 1 - asym_params['lambda_up'] * (vol_level - 1)
    vol_adj_rising = np.clip(vol_adj_rising, 0.5, 1.5)
    beta_rising_adj = beta_rising * vol_adj_rising
    
    ax1.plot(ff_range, beta_rising_adj, color=color, linewidth=2.5, label=vol_label)

ax1.axhline(y=asym_params['beta_min'], color='gray', linestyle='--', alpha=0.5, 
            label='Beta Min (' + str(int(asym_params["beta_min"]*100)) + '%)')
ax1.axhline(y=asym_params['beta_max'], color='gray', linestyle='--', alpha=0.5,
            label='Beta Max (' + str(int(asym_params["beta_max"]*100)) + '%)')
ax1.axvline(x=asym_params['m'], color='purple', linestyle=':', alpha=0.7, 
            label='Inflection (m=' + f'{asym_params["m"]:.2f}' + '%)')

ax1.set_xlabel('Fed Funds Rate (%)', fontsize=11)
ax1.set_ylabel('Deposit Beta (Pass-Through)', fontsize=11)
ax1.set_xlim(0, 6)
ax1.set_ylim(0.3, 0.8)
ax1.legend(loc='lower right', fontsize=9)
ax1.grid(True, alpha=0.3)
lambda_up_text = r'$\lambda_{up}$' + f' = {asym_params["lambda_up"]:.3f}'
ax1.text(0.05, 0.95, lambda_up_text, transform=ax1.transAxes, fontsize=11, 
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Panel B: Falling Rate Environment
ax2 = axes[1]
ax2.set_title('Panel B: Falling Rate Environment', fontsize=12, fontweight='bold')

for vol_level, vol_label, color in [(0.5, 'Low Vol (0.5x)', '#2ecc71'), 
                                     (1.0, 'Normal Vol (1.0x)', '#3498db'),
                                     (2.0, 'High Vol (2.0x)', '#e74c3c')]:
    # Falling rates: use lambda_down
    beta_falling = model.logistic_beta(ff_range, asym_params['k'], asym_params['m'], 
                                        asym_params['beta_min'], asym_params['beta_max'])
    vol_adj_falling = 1 - asym_params['lambda_down'] * (vol_level - 1)
    vol_adj_falling = np.clip(vol_adj_falling, 0.5, 1.5)
    beta_falling_adj = beta_falling * vol_adj_falling
    
    ax2.plot(ff_range, beta_falling_adj, color=color, linewidth=2.5, label=vol_label)

ax2.axhline(y=asym_params['beta_min'], color='gray', linestyle='--', alpha=0.5,
            label='Beta Min (' + str(int(asym_params["beta_min"]*100)) + '%)')
ax2.axhline(y=asym_params['beta_max'], color='gray', linestyle='--', alpha=0.5,
            label='Beta Max (' + str(int(asym_params["beta_max"]*100)) + '%)')
ax2.axvline(x=asym_params['m'], color='purple', linestyle=':', alpha=0.7,
            label='Inflection (m=' + f'{asym_params["m"]:.2f}' + '%)')

ax2.set_xlabel('Fed Funds Rate (%)', fontsize=11)
ax2.set_ylabel('Deposit Beta (Pass-Through)', fontsize=11)
ax2.set_xlim(0, 6)
ax2.set_ylim(0.3, 0.8)
ax2.legend(loc='lower right', fontsize=9)
ax2.grid(True, alpha=0.3)
lambda_down_text = r'$\lambda_{down}$' + f' = {asym_params["lambda_down"]:.3f}'
ax2.text(0.05, 0.95, lambda_down_text, transform=ax2.transAxes, fontsize=11,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

plt.suptitle('Figure 3: Asymmetric Dynamic Beta Evolution\nVolatility Dampening Differs by Rate Direction', 
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('outputs/figures/figure3_asymmetric_beta_evolution.png', dpi=300, bbox_inches='tight')
plt.savefig('outputs/figures/figure3_asymmetric_beta_evolution.pdf', bbox_inches='tight')
print("   Saved: outputs/figures/figure3_asymmetric_beta_evolution.png")
print("   Saved: outputs/figures/figure3_asymmetric_beta_evolution.pdf")
plt.close()

# Save key results for paper update
print("\n" + "=" * 80)
print("KEY RESULTS FOR PAPER UPDATE:")
print("=" * 80)
print("""
TABLE 4 UPDATE (Asymmetric Vol-Adjusted Model Parameters):
----------------------------------------------------------""")
print(f"Parameter      | Estimate  | Interpretation")
print(f"---------------|-----------|----------------------------------")
print(f"alpha          | {asym_params['alpha']:.4f}    | Base margin when Fed Funds = 0%")
print(f"k              | {asym_params['k']:.4f}    | Transition speed")
print(f"m              | {asym_params['m']:.4f}    | Inflection at {asym_params['m']:.2f}% Fed Funds")
print(f"beta_min       | {asym_params['beta_min']:.4f}    | {asym_params['beta_min']*100:.0f}% minimum sensitivity")
print(f"beta_max       | {asym_params['beta_max']:.4f}    | {asym_params['beta_max']*100:.0f}% maximum sensitivity")
print(f"gamma_fhlb     | {asym_params['gamma_fhlb']:.4f}    | FHLB spread coefficient")
print(f"gamma_term     | {asym_params['gamma_term']:.4f}   | Term spread coefficient")
print(f"lambda_up      | {asym_params['lambda_up']:.4f}    | {asym_params['lambda_up']*100:.1f}% dampening (rising rates)")
print(f"lambda_down    | {asym_params['lambda_down']:.4f}    | {asym_params['lambda_down']*100:.1f}% dampening (falling rates)")
print(f"""
KEY INSIGHT: Banks delay competitive repricing MORE aggressively when rates 
are rising ({asym_params['lambda_up']*100:.1f}% dampening) than when falling ({asym_params['lambda_down']*100:.1f}% dampening).
This state-dependent inertia protects margins during hiking cycles.
""")

print("\nAnalysis complete!")
