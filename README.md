# MMDA Dynamic Beta Repricing Model

**A Volatility-Adjusted Framework for Money Market Deposit Account Rate Sensitivity**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Author:** Chih L. Chen, BTRM, CFA, FRM

## Overview

This repository contains the complete implementation and documentation for a dynamic beta model designed to estimate Money Market Deposit Account (MMDA) rate sensitivity to changes in policy interest rates. Unlike traditional static beta approaches, this model captures the nonlinear, rate-dependent nature of deposit competition that was dramatically exposed during the Federal Reserve's 2022-2025 tightening cycle.

### Key Features

- **Dynamic Beta Estimation:** Logistic framework capturing how deposit betas evolve with interest rate levels
- **Volatility Adjustment:** Accounts for reduced pass-through during volatile rate environments  
- **Full Reproducibility:** All results traceable to source data with complete output exports
- **SR11-7 Compliant Documentation:** Model development document meeting regulatory standards
- **Comprehensive Validation:** Challenger models, diagnostic tests, and out-of-sample performance

### Model Performance

| Metric | Value |
|--------|-------|
| R² | 0.9858 |
| RMSE | 0.1077% |
| Out-of-Sample RMSE (2022-2025) | 0.1180% |
| Improvement vs. Static Beta | 37% lower forecast errors |

## Quick Start

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scipy statsmodels scikit-learn openpyxl
```

### Run the Analysis

```bash
python run_analysis_with_outputs.py
```

This will generate all outputs in the `outputs/` directory:
- CSV/Excel files with all model results
- PNG visualizations
- Text reports

### View Results

After running, check:
- `outputs/model_results/` - Parameter estimates, performance metrics, predictions
- `outputs/visualizations/` - Charts and diagnostic plots
- `outputs/reports/` - Executive summary and detailed analysis report

## Repository Structure

```
NMD_Beta/
├── README.md                              # This file
├── LICENSE                                # MIT License
├── requirements.txt                       # Python dependencies
├── bankratemma.csv                        # Source data (Bloomberg/Bankrate/FRED)
├── mmda_dynamic_beta_model_complete.py    # Core model implementation
├── run_analysis_with_outputs.py           # Reproducible analysis script
├── generate_model_document.py             # Auto-generate model documentation
├── mmda-dynamic-beta-model-dev-doc.md     # Full model documentation (SR11-7 format)
├── mmda-dynamic-beta-academic-paper.md    # Academic paper version (APA format)
├── config/
│   └── document_config.json               # Configuration for document generator
├── templates/
│   └── model_development_document_template.md  # Document template
└── outputs/                               # Generated outputs
    ├── data/                              # Processed data files
    ├── model_results/                     # Parameter estimates, metrics
    ├── visualizations/                    # Charts and plots
    ├── reports/                           # Analysis reports
    └── *.pdf                              # Generated PDF documents
```

## Model Specification

### Dynamic Beta Function

The model uses a volatility-adjusted logistic framework:

```
β_t = [β_min + (β_max - β_min) / (1 + exp(-k × (r_t - m)))] × (1 - λ × σ_t/σ*)
```

Where:
- `β_min`, `β_max`: Deposit sensitivity bounds (40% to 70%)
- `m`: Competition inflection point (2.99% Fed Funds)
- `k`: Transition steepness (0.59)
- `λ`: Volatility dampening factor (22.4%)

### Complete Specification

```
MMDA_Rate_t = α + β_t × FedFunds_t + γ₁ × FHLB_Spread_t + γ₂ × Term_Spread_t + ε_t
```

## Data Sources

| Variable | Source | Description |
|----------|--------|-------------|
| ILMDHYLD | Bloomberg/Bankrate | High-yield MMDA rate benchmark |
| FEDL01 | FRED/Bloomberg | Federal Funds Effective Rate |
| FHLK3MSPRD | Bloomberg | FHLB vs SOFR 3M liquidity premium |
| 1Y_3M_SPRD | Bloomberg | 1Y-3M SOFR OIS term spread |

**Sample Period:** January 2017 - March 2025 (99 monthly observations)

## Use Cases

This model is designed for bank ALM practitioners working on:

- **NII Sensitivity Analysis:** Understanding how deposit costs respond to rate scenarios
- **EVE Calculations:** Duration estimation for non-maturity deposit portfolios
- **Regulatory Stress Testing:** CCAR, DFAST, Basel III IRRBB
- **FTP Calibration:** Pricing interest rate risk in deposit products
- **Strategic Planning:** Balance sheet optimization decisions

## Limitations

- Calibrated on 2017-2025 data; extrapolation beyond 6% Fed Funds requires caution
- Monthly frequency may miss intra-month dynamics
- All models exhibit residual autocorrelation (common in financial time series)
- Model assumes stable competitive structure; may not capture fintech disruption

See the full [model documentation](mmda-dynamic-beta-model-dev-doc.md) for comprehensive discussion of assumptions and limitations.

## Document Generation

The repository includes a template-based document generator for creating SR11-7 compliant model documentation:

### Generate Model Development Document

```bash
# Generate with default settings
python generate_model_document.py

# Generate with custom configuration
python generate_model_document.py --config config/document_config.json --output outputs/Model_Development_Document.md

# Create a sample configuration file
python generate_model_document.py --create-config
```

### Customize the Template

1. Edit `config/document_config.json` to set your organization details
2. Modify `templates/model_development_document_template.md` for structural changes
3. Run the generator to produce updated documentation

This approach allows you to:
- Automatically populate tables from model outputs
- Maintain consistent formatting across updates
- Easily adapt for different models or organizations

## Citation

If you use this model in academic research or publications, please cite:

```
Chen, C. L. (2026). MMDA Dynamic Beta Repricing Model: A Volatility-Adjusted 
Framework for Deposit Rate Sensitivity. GitHub Repository.
https://github.com/deechean/NMD_Beta
```

## Acknowledgments

This model was developed with assistance from AI systems (Perplexity Labs, Google Gemini Pro 2.5, Anthropic Claude) for computational support, code development, and documentation preparation. All economic reasoning, model specification choices, and business application recommendations reflect the professional judgment of the model owner. See the acknowledgments section in the model documentation for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or collaboration opportunities, please open an issue in this repository.

---

*This model is provided for educational and research purposes. Users should perform their own validation before applying to production risk management applications.*
