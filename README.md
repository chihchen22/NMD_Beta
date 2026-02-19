# Dynamic Deposit Betas: An Asymmetric Volatility-Adjusted S-Curve Framework for MMDA Rate Sensitivity

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![SSRN](https://img.shields.io/badge/SSRN-6269838-blue)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6269838)

**Author:** Chih L. Chen, BTRM, CFA, FRM

**Paper:** [Dynamic Deposit Betas: An Asymmetric Volatility-Adjusted S-Curve Framework for MMDA Rate Sensitivity](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6269838) (SSRN, February 2026)

## Overview

This repository implements an 8-parameter dynamic beta model for estimating Money Market Deposit Account (MMDA) rate sensitivity to policy rate changes. The model captures the nonlinear, rate-dependent nature of deposit competition exposed during the Fed’s 2022-2025 tightening cycle.

The core specification combines a logistic S-curve (rate-level-dependent beta), asymmetric volatility dampening (distinct behavior in rising vs. falling rate environments), autoregressive smoothing (bounded month-to-month beta changes), and a Nerlove partial adjustment filter for scenario forecasting. Standard errors use the Huber-White sandwich estimator with Newey-West HAC weighting.

### Key Results

| Metric | Value |
|--------|-------|
| In-sample R² | 0.9870 |
| RMSE | 10.3 bps |
| Parameters | 8 |
| Partial adjustment speed (θ) | 0.47 |
| Asymmetric dampening | λ_up = 25.5%, λ_down = 22.3% (p < 0.001) |

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Reproduce All Results

```bash
python regenerate_all_outputs.py      # Core model comparison (14 files)
python scenario_shock_analysis.py     # Parallel shock scenarios with partial adjustment
python paper_enhancements.py          # OOS validation, duration, portfolio example, bootstrap
python regenerate_figures.py          # Data dashboard and visualization figures
```

### Generate Paper PDF

Requires [pandoc](https://pandoc.org/) and a LaTeX distribution (e.g., MiKTeX with XeLaTeX):

```bash
python convert_to_pdf.py
```

## Repository Structure

```
NMD_Beta/
├── README.md
├── LICENSE
├── requirements.txt
├── bankratemma.csv                        # Source data (Jan 2017 - Mar 2025, n=99)
│
├── enhanced_dynamic_beta_model.py         # Core model: AR smoothing + sandwich SEs
├── mmda_dynamic_beta_model_complete.py    # Full model framework class
├── two_regime_ecm_vs_ml.py               # 2-regime ECM challenger model
│
├── regenerate_all_outputs.py             # Master output generator (14 files)
├── scenario_shock_analysis.py            # Scenario analysis with partial adjustment
├── paper_enhancements.py                 # OOS, robustness, duration, portfolio, bootstrap
├── regenerate_figures.py                 # Data dashboard and core visualizations
├── regenerate_figures_with_ols.py        # Model fit comparison chart
├── run_asymmetric_analysis.py            # Asymmetric beta evolution chart
│
├── mmda-dynamic-beta-academic-paper-v2.md   # Paper source (Markdown)
├── convert_to_pdf.py                        # Markdown to PDF via pandoc + XeLaTeX
├── pandoc_preamble.tex                      # LaTeX preamble for PDF styling
│
└── outputs/
    ├── v2_comparison/                    # Model comparison outputs (14 files)
    ├── scenario_analysis/                # Shock scenario results (4 files)
    ├── paper_enhancements/               # OOS, duration, portfolio, bootstrap figures
    ├── figures/                          # Asymmetric beta and model charts
    └── visualizations/                   # Data dashboard and fit comparison
```

## Model Specification

The model uses an asymmetric volatility-adjusted logistic beta with AR smoothing to prevent volatile month-to-month jumps. For forward-looking scenarios, a Nerlove partial adjustment filter governs the transition path, where roughly half the gap between the current deposit rate and its model-implied equilibrium closes each month. See the [paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6269838) for the full specification.

### Estimated Parameters

| Parameter | Description | Value |
|-----------|-------------|-------|
| α | Intercept | 0.073 |
| k | Logistic steepness | 0.566 |
| m | Inflection point | 3.919% |
| β_min | Lower beta bound | 0.433 |
| β_max | Upper beta bound | 0.800 |
| γ_fhlb | FHLB spread coefficient | 1.049 |
| λ_up | Rising-rate dampening | 0.255 |
| λ_down | Falling-rate dampening | 0.223 |

## Data Sources

| Variable | Source | Description |
|----------|--------|-------------|
| ILMDHYLD | Bloomberg/Bankrate | High-yield MMDA rate benchmark |
| FEDL01 | FRED/Bloomberg | Federal Funds Effective Rate |
| FHLK3MSPRD | Bloomberg | FHLB vs SOFR 3M liquidity premium |

**Sample Period:** January 2017 - March 2025 (99 monthly observations)

## Use Cases

- **NII Sensitivity Analysis:** Deposit cost response to rate scenarios
- **EVE Calculations:** Duration estimation for non-maturity deposit portfolios
- **Regulatory Stress Testing:** CCAR, DFAST, Basel III IRRBB
- **FTP Calibration:** Pricing interest rate risk in deposit products
- **Custom Rate Index:** Deploy as a deposit rate forecasting index in ALM platforms

## Citation

```bibtex
@article{chen2026dynamic,
  title={Dynamic Deposit Betas: An Asymmetric Volatility-Adjusted S-Curve
         Framework for MMDA Rate Sensitivity},
  author={Chen, Chih L.},
  journal={SSRN Electronic Journal},
  year={2026},
  doi={10.2139/ssrn.6269838},
  url={https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6269838}
}
```

## Acknowledgments

This model was developed with assistance from AI systems (Perplexity Labs, Google Gemini Pro 2.5, Anthropic Claude) for computational support, code development, and documentation preparation. All economic reasoning, model specification choices, and business application recommendations reflect the professional judgment of the author.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

*This model is provided for educational and research purposes. Users should perform their own validation before applying to production risk management applications.*
