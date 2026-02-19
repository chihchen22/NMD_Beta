
"""Two-regime nonlinear-beta (S-curve) long-run + ECM, with challengers.

Implements what you asked for:
1) Recommended 2-regime approach
   - For each regime, select a sample window that best supports residual stationarity
     (cointegration-friendly) while keeping a nonlinear S-curve beta in the long run.
   - Fit long-run: y_t = alpha + beta(r_t)*r_t + gamma'X_t + e_t
   - If residuals screen as I(0), fit ECM using e_{t-1}

2) Challenger models
   A) "Traditional econometric" ECM variants (still dynamic beta):
      - Symmetric ECM: Δy_t ~ c + λ e_{t-1} + Δr + ΔTS + ΔFS + optional lags
      - Asymmetric ECM: split Δr into Δr+ and Δr-

   B) ML challenger for dynamic beta and lagged repricing:
      - Fit a time-series ML model to predict Δy_t using:
        ECT_{t-1}, Δr_t, lagged Δr, level r, spreads, vol, and lagged Δy.
      - Uses scikit-learn HistGradientBoostingRegressor.

Evaluation:
- For each regime, we do a simple within-regime holdout:
  train = all but last 12 months, test = last 12 months.
  Forecast recursively in levels using predicted Δy.

Notes (validation):
- Residual-based ADF/KPSS in nonlinear cointegration is a SCREEN unless bootstrapped.
- This script prioritizes a practical end-to-end pipeline you can iterate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.stattools import adfuller, kpss

from sklearn.ensemble import HistGradientBoostingRegressor


# -----------------------
# Data config
# -----------------------
START_MIN = "2014-01-01"
END_MAX = "2024-12-31"

TERM_SPREAD_CANDIDATES = [
    # Classic term spreads (truncate around 2022 in this dataset)
    "10Y_3M_SPRD",
    "5Y_3M_SPRD",
    "3Y_3M_SPRD",
    "2Y_3M_SPRD",
    # Short-end / money-market spreads (better post-2022 coverage here)
    "1Y_3M_SPRD",
    "6M_3M_SPRD",
    "3M_1M_SPRD",
]
FHLB_SPREAD_CANDIDATES = ["BVCSUB3MSPRD", "FHLK3MSPRD"]

TEST_MONTHS = 12


# -----------------------
# Regime definitions
# -----------------------
REGIMES = {
    "Regime1_pre2019": {
        "start_candidates": ["2014-01-01", "2015-01-01", "2016-01-01"],
        "end_candidates": ["2018-12-31", "2019-12-31"],
        # Need enough length for (train + holdout). Spreads often truncate 2019.
        "min_obs": 48,
    },
    "Regime2_post2022": {
        "start_candidates": ["2020-01-01", "2021-01-01", "2022-01-01"],
        "end_candidates": ["2024-12-31", "2023-12-31"],
        # Allow shorter windows here; cointegration tests are low power, but we need feasibility.
        "min_obs": 36,
    },
}

MAX_DY_LAGS = 6
MAX_DX_LAGS = 6


# =============================================================================
# Helpers
# =============================================================================

def beta_gompertz(r: np.ndarray, k: float, m: float, beta_min: float, beta_max: float) -> np.ndarray:
    beta = beta_min + (beta_max - beta_min) * np.exp(-np.exp(-k * (r - m)))
    return np.clip(beta, beta_min, beta_max)


def safe_adf_p(x: pd.Series, regression: str) -> float:
    x = x.dropna()
    stat, p, *_ = adfuller(x, autolag="AIC", regression=regression)
    return float(p)


def safe_kpss_p(x: pd.Series, regression: str) -> float:
    x = x.dropna()
    stat, p, *_ = kpss(x, regression=regression, nlags="auto")
    return float(p)


def ewma_vol_excess(dr: pd.Series, span: int = 24) -> pd.Series:
    vol = dr.ewm(span=span).std().fillna(0.0)
    mu = float(vol.mean())
    if not np.isfinite(mu) or mu <= 0:
        mu = 1.0
    return vol / mu - 1.0


def coerce_numeric(series: pd.Series) -> pd.Series:
    """Coerce messy numeric strings to floats.

    Handles formats observed in bank exports, e.g. " (0.04)" for -0.04.
    """
    if series is None:
        return series
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    s = series.astype(str).str.strip()
    s = s.str.replace(",", "", regex=False)
    # Parentheses as negatives
    s = s.str.replace(r"^\((.*)\)$", r"-\1", regex=True)
    # Remove stray parentheses/spaces
    s = s.str.replace("(", "", regex=False).str.replace(")", "", regex=False)
    s = s.str.replace(" ", "", regex=False)
    return pd.to_numeric(s, errors="coerce")


def build_lag_matrix(x: pd.Series, max_lag: int, prefix: str) -> pd.DataFrame:
    return pd.concat({f"{prefix}_L{lag}": x.shift(lag) for lag in range(1, max_lag + 1)}, axis=1)


def finite_or_zero(x: float | int) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    return v if np.isfinite(v) else 0.0


def select_ecm_lags_by_aic(
    df_reg: pd.DataFrame,
    y_col: str,
    x_cols: list[str],
    min_rows: int,
    max_p: int = MAX_DY_LAGS,
    max_q: int = MAX_DX_LAGS,
) -> tuple[int, int, sm.regression.linear_model.RegressionResultsWrapper, list[str]] | None:
    best = None
    best_aic = np.inf
    max_p = int(max(0, max_p))
    max_q = int(max(0, max_q))
    for p in range(0, max_p + 1):
        for q in range(0, max_q + 1):
            cols = []
            cols += [c for c in df_reg.columns if c.startswith("dY_L") and int(c.split("L")[1]) <= p]
            cols += [c for c in df_reg.columns if c.startswith("dR_L") and int(c.split("L")[1]) <= q]
            cols += [c for c in df_reg.columns if c.startswith("dTS_L") and int(c.split("L")[1]) <= q]
            cols += [c for c in df_reg.columns if c.startswith("dFS_L") and int(c.split("L")[1]) <= q]
            cols += [c for c in x_cols if c in df_reg.columns]

            tmp = df_reg[[y_col] + cols].dropna().copy()
            if len(tmp) < min_rows:
                continue

            X = sm.add_constant(tmp[cols])
            res = sm.OLS(tmp[y_col], X).fit()
            if res.aic < best_aic:
                best_aic = float(res.aic)
                best = (p, q, res, cols)
    return best


@dataclass
class LongRunParams:
    alpha: float
    k: float
    m: float
    beta_min: float
    beta_max: float
    gamma_ts: float
    gamma_fs: float


@dataclass
class WindowChoice:
    start: str
    end: str
    term_col: str
    fhlb_col: str
    n: int
    rmse: float
    adf_n_p: float
    kpss_c_p: float
    params: LongRunParams
    flags: str

    @property
    def passes_gate(self) -> bool:
        return (self.adf_n_p < 0.05) and (self.kpss_c_p > 0.05)

    @property
    def score(self) -> float:
        # Bigger is better.
        adf = max(self.adf_n_p, 1e-12)
        kpss = max(min(self.kpss_c_p, 1.0), 1e-12)
        return (-math.log10(adf)) + (math.log10(kpss)) - 0.5 * math.log10(max(self.rmse, 1e-6))


def fit_long_run(df_fit: pd.DataFrame, term_col: str, fhlb_col: str) -> tuple[LongRunParams, pd.Series, pd.Series, float, str] | None:
    tmp = df_fit.copy()
    for c in ["Rate", "Rate_Paid", term_col, fhlb_col]:
        if c in tmp.columns:
            tmp[c] = coerce_numeric(tmp[c])
    tmp = tmp.dropna(subset=["Rate", "Rate_Paid", term_col, fhlb_col]).copy()
    if len(tmp) < 24:
        return None

    r = tmp["Rate"].to_numpy(dtype=float)
    y = tmp["Rate_Paid"].to_numpy(dtype=float)
    ts = tmp[term_col].to_numpy(dtype=float)
    fs = tmp[fhlb_col].to_numpy(dtype=float)

    def sse(p: np.ndarray) -> float:
        alpha, k, m, beta_min, beta_max, gamma_ts, gamma_fs = p
        if beta_min >= beta_max or k <= 0:
            return 1e18
        beta = beta_gompertz(r, k, m, beta_min, beta_max)
        yhat = alpha + beta * r + gamma_ts * ts + gamma_fs * fs
        resid = y - yhat
        return float(np.sum(resid**2))

    bounds = [
        (-1.0, 1.5),
        (0.05, 6.0),
        (0.0, 8.0),
        (0.05, 0.80),
        (0.10, 0.99),
        (-1.5, 1.5),
        (-1.5, 1.5),
    ]

    starts = [
        [0.30, 1.0, 2.0, 0.30, 0.55, 0.10, 0.00],
        [0.20, 0.7, 3.0, 0.25, 0.50, 0.10, 0.10],
        [0.35, 1.5, 1.5, 0.35, 0.65, 0.00, 0.10],
        [0.10, 2.0, 2.0, 0.20, 0.45, 0.20, -0.10],
        # Extra starts for shorter/volatile windows
        [0.50, 0.8, 4.0, 0.15, 0.80, 0.00, 0.00],
        [0.00, 1.2, 1.0, 0.10, 0.40, 0.20, 0.20],
    ]

    best = None
    best_val = np.inf
    for start in starts:
        res = minimize(sse, start, bounds=bounds, method="L-BFGS-B", options={"maxiter": 50000})
        if float(res.fun) < best_val:
            best_val = float(res.fun)
            best = res

    if best is None:
        return None

    # On short windows, L-BFGS-B can hit iteration limits while still landing
    # on a usable optimum. Treat that as usable if the objective is finite.
    opt_failed = (not bool(best.success))
    if opt_failed and (not np.isfinite(best.fun)):
        return None

    alpha, k, m, beta_min, beta_max, gamma_ts, gamma_fs = map(float, best.x)
    beta = beta_gompertz(r, k, m, beta_min, beta_max)
    y_star = alpha + beta * r + gamma_ts * ts + gamma_fs * fs

    resid = pd.Series(y - y_star, index=tmp.index, name="e")
    y_star_s = pd.Series(y_star, index=tmp.index, name="y_star")
    rmse = float(np.sqrt(np.mean((y - y_star) ** 2)))

    flags = []
    if opt_failed:
        flags.append("opt_not_converged")
    if abs(k - 6.0) < 1e-3:
        flags.append("k@ub")
    if abs(beta_max - 0.99) < 1e-3:
        flags.append("beta_max@ub")

    return LongRunParams(alpha, k, m, beta_min, beta_max, gamma_ts, gamma_fs), resid, y_star_s, rmse, ",".join(flags)


def choose_window(df: pd.DataFrame, start_candidates: list[str], end_candidates: list[str], min_obs: int) -> WindowChoice:
    choices: list[WindowChoice] = []
    seen: set[tuple[str, str, str, str]] = set()

    for start in start_candidates:
        for end in end_candidates:
            if pd.to_datetime(end) <= pd.to_datetime(start):
                continue
            df_win = df.loc[(df.index >= start) & (df.index <= end)].copy()
            if len(df_win) < (min_obs + TEST_MONTHS):
                continue

            for term_col in TERM_SPREAD_CANDIDATES:
                for fhlb_col in FHLB_SPREAD_CANDIDATES:
                    if term_col not in df_win.columns or fhlb_col not in df_win.columns:
                        continue

                    # Require the chosen columns to exist and be non-missing across the window.
                    df_win2 = df_win.dropna(subset=["Rate", "Rate_Paid", term_col, fhlb_col]).copy()
                    if len(df_win2) < (min_obs + TEST_MONTHS):
                        continue

                    # selection is done on train portion to reduce look-ahead
                    df_train = df_win2.iloc[: max(0, len(df_win2) - TEST_MONTHS)].copy()
                    if len(df_train) < min_obs:
                        continue

                    out = fit_long_run(df_train, term_col, fhlb_col)
                    if out is None:
                        continue
                    params, resid, _, rmse, flags = out

                    adf_n_p = safe_adf_p(resid, regression="n")
                    kpss_c_p = safe_kpss_p(resid, regression="c")

                    key = (str(df_win2.index.min().date()), str(df_win2.index.max().date()), term_col, fhlb_col)
                    if key in seen:
                        continue
                    seen.add(key)

                    choices.append(
                        WindowChoice(
                            start=key[0],
                            end=key[1],
                            term_col=term_col,
                            fhlb_col=fhlb_col,
                            n=len(df_train),
                            rmse=rmse,
                            adf_n_p=adf_n_p,
                            kpss_c_p=kpss_c_p,
                            params=params,
                            flags=flags,
                        )
                    )

    if not choices:
        raise RuntimeError("No feasible windows could be fit")

    passing = [c for c in choices if c.passes_gate]
    ranked = sorted(choices, key=lambda c: c.score, reverse=True)
    best = sorted(passing, key=lambda c: c.score, reverse=True)[0] if passing else ranked[0]

    print("\nTop window candidates:")
    for c in ranked[:5]:
        tag = "PASS" if c.passes_gate else "----"
        print(
            f"  {tag} {c.start}..{c.end} | TS={c.term_col} FS={c.fhlb_col} | train_n={c.n} | RMSE={c.rmse:.4f} | ADFn={c.adf_n_p:.4f} KPSS={c.kpss_c_p:.4f} | {c.flags}"
        )

    return best


def fit_ecm(
    df_train: pd.DataFrame,
    term_col: str,
    fhlb_col: str,
    lr: LongRunParams,
    asymmetric: bool,
    max_p: int = MAX_DY_LAGS,
    max_q: int = MAX_DX_LAGS,
) -> tuple[sm.regression.linear_model.RegressionResultsWrapper, list[str]]:
    tmp = df_train.dropna(subset=["Rate", "Rate_Paid", term_col, fhlb_col]).copy()
    r = tmp["Rate"].to_numpy(dtype=float)

    beta = beta_gompertz(r, lr.k, lr.m, lr.beta_min, lr.beta_max)
    y_star = lr.alpha + beta * r + lr.gamma_ts * tmp[term_col].to_numpy(dtype=float) + lr.gamma_fs * tmp[fhlb_col].to_numpy(dtype=float)
    e = pd.Series(tmp["Rate_Paid"].to_numpy(dtype=float) - y_star, index=tmp.index)

    reg = pd.DataFrame(index=tmp.index)
    reg["dY"] = tmp["Rate_Paid"].diff()
    reg["ECT_L1"] = e.shift(1)

    reg["dR"] = tmp["Rate"].diff()
    reg["dTS"] = tmp[term_col].diff()
    reg["dFS"] = tmp[fhlb_col].diff()

    if asymmetric:
        reg["dR_pos"] = reg["dR"].clip(lower=0.0)
        reg["dR_neg"] = reg["dR"].clip(upper=0.0)

    reg = pd.concat(
        [
            reg,
            build_lag_matrix(reg["dY"], MAX_DY_LAGS, "dY"),
            build_lag_matrix(reg["dR"], MAX_DX_LAGS, "dR"),
            build_lag_matrix(reg["dTS"], MAX_DX_LAGS, "dTS"),
            build_lag_matrix(reg["dFS"], MAX_DX_LAGS, "dFS"),
        ],
        axis=1,
    )

    x_cols = ["ECT_L1", "dTS", "dFS"]
    if asymmetric:
        x_cols += ["dR_pos", "dR_neg"]
    else:
        x_cols += ["dR"]

    # Need a feasible threshold for short regime samples.
    # After differencing + lags, usable rows can drop quickly.
    min_rows = max(18, min(36, len(reg) - 1))
    best = select_ecm_lags_by_aic(reg, y_col="dY", x_cols=x_cols, min_rows=min_rows, max_p=max_p, max_q=max_q)
    if best is None:
        raise RuntimeError("ECM lag selection failed")

    p, q, res, cols = best

    # For reporting/inference, use HAC as default since ARCH is common here.
    res_hac = res.get_robustcov_results(cov_type="HAC", maxlags=6)

    print(f"\nECM {'asymmetric' if asymmetric else 'symmetric'} selected lags: Δy up to {p}, Δx up to {q}")
    print(res_hac.summary())

    return res_hac, cols


def forecast_ecm(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    term_col: str,
    fhlb_col: str,
    lr: LongRunParams,
    ecm_res,
    cols: list[str],
    asymmetric: bool,
    recursive: bool = False,
) -> pd.Series:
    """Forecast level y using fitted ECM on Δy.

    Validation default is one-step-ahead (recursive=False): uses actual lagged y, Δy.
    For scenario simulation, use recursive=True.
    """
    full = pd.concat([df_train, df_test]).copy()
    for c in ["Rate", term_col, fhlb_col, "Rate_Paid"]:
        if c in full.columns:
            full[c] = coerce_numeric(full[c])
    # Allow Rate_Paid to be missing in df_test (scenario forecasting).
    # We only require exogenous drivers to be present.
    full = full.dropna(subset=["Rate", term_col, fhlb_col]).copy()

    # Precompute long-run y* based on observed regressors
    r = full["Rate"].to_numpy(dtype=float)
    beta = beta_gompertz(r, lr.k, lr.m, lr.beta_min, lr.beta_max)
    y_star = lr.alpha + beta * r + lr.gamma_ts * full[term_col].to_numpy(dtype=float) + lr.gamma_fs * full[fhlb_col].to_numpy(dtype=float)
    y_star = pd.Series(y_star, index=full.index)

    # Storage
    yhat = pd.Series(index=full.index, dtype=float)

    # Initialize yhat
    if recursive:
        last_train_date = df_train.index.max()
        yhat.loc[:last_train_date] = full.loc[:last_train_date, "Rate_Paid"]

    # Prepare Δ series
    dR = full["Rate"].diff()
    dTS = full[term_col].diff()
    dFS = full[fhlb_col].diff()

    # Forecast over test dates
    test_dates = df_test.index
    for t in test_dates:
        if t not in full.index:
            continue
        t_prev = full.index[full.index.get_loc(t) - 1]

        # Build a single-row regressor frame matching cols
        row = {}
        # ECT uses previous period y (actual for one-step; yhat for recursive)
        y_prev = float(yhat.loc[t_prev]) if recursive else float(full.loc[t_prev, "Rate_Paid"])
        ect = float(y_prev - y_star.loc[t_prev])
        row["ECT_L1"] = ect

        # contemporaneous diffs
        row["dTS"] = finite_or_zero(dTS.loc[t])
        row["dFS"] = finite_or_zero(dFS.loc[t])

        if asymmetric:
            dr = finite_or_zero(dR.loc[t])
            row["dR_pos"] = max(0.0, dr)
            row["dR_neg"] = min(0.0, dr)
        else:
            row["dR"] = finite_or_zero(dR.loc[t])

        # lag terms selected
        for c in cols:
            if c.startswith("dY_L"):
                lag = int(c.split("L")[1])
                # Δy lag = y_{t-1} - y_{t-2}, etc. (actual for one-step)
                idx = full.index.get_loc(t)
                if idx - lag < 1:
                    row[c] = 0.0
                else:
                    t_lag = full.index[idx - lag]
                    t_lag_prev = full.index[idx - lag - 1]
                    if recursive:
                        row[c] = float(yhat.loc[t_lag] - yhat.loc[t_lag_prev])
                    else:
                        row[c] = float(full.loc[t_lag, "Rate_Paid"] - full.loc[t_lag_prev, "Rate_Paid"])
            elif c.startswith("dR_L"):
                lag = int(c.split("L")[1])
                row[c] = finite_or_zero(dR.shift(lag).loc[t])
            elif c.startswith("dTS_L"):
                lag = int(c.split("L")[1])
                row[c] = finite_or_zero(dTS.shift(lag).loc[t])
            elif c.startswith("dFS_L"):
                lag = int(c.split("L")[1])
                row[c] = finite_or_zero(dFS.shift(lag).loc[t])

        X = pd.DataFrame([row])
        X = sm.add_constant(X, has_constant="add")
        # Align columns to model
        X = X.reindex(columns=ecm_res.model.exog_names, fill_value=0.0)

        dy_hat = float(ecm_res.predict(X)[0])
        if recursive:
            yhat.loc[t] = float(yhat.loc[t_prev] + dy_hat)
        else:
            yhat.loc[t] = float(full.loc[t_prev, "Rate_Paid"] + dy_hat)

    return yhat.loc[df_test.index]


def fit_ml_delta_model(df_train: pd.DataFrame, term_col: str, fhlb_col: str, lr: LongRunParams):
    tmp = df_train.dropna(subset=["Rate", "Rate_Paid", term_col, fhlb_col]).copy()

    # Build ECT using long-run model
    r = tmp["Rate"].to_numpy(dtype=float)
    beta = beta_gompertz(r, lr.k, lr.m, lr.beta_min, lr.beta_max)
    y_star = lr.alpha + beta * r + lr.gamma_ts * tmp[term_col].to_numpy(dtype=float) + lr.gamma_fs * tmp[fhlb_col].to_numpy(dtype=float)
    e = pd.Series(tmp["Rate_Paid"].to_numpy(dtype=float) - y_star, index=tmp.index)

    dY = tmp["Rate_Paid"].diff()
    dR = tmp["Rate"].diff()

    vol_excess = ewma_vol_excess(dR)

    feats = pd.DataFrame(index=tmp.index)
    feats["ECT_L1"] = e.shift(1)
    feats["dR"] = dR
    feats["dR_L1"] = dR.shift(1)
    feats["Rate"] = tmp["Rate"]
    feats["TS"] = tmp[term_col]
    feats["FS"] = tmp[fhlb_col]
    feats["dTS"] = tmp[term_col].diff()
    feats["dFS"] = tmp[fhlb_col].diff()
    feats["vol_excess"] = vol_excess
    feats["dY_L1"] = dY.shift(1)

    # dynamic beta features
    feats["beta"] = beta
    feats["x_eff"] = beta * tmp["Rate"].to_numpy(dtype=float)
    feats["dx_eff"] = feats["x_eff"].diff()

    target = dY

    data = pd.concat([target.rename("dY"), feats], axis=1).dropna().copy()

    X = data.drop(columns=["dY"])
    y = data["dY"].to_numpy(dtype=float)

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        max_depth=3,
        learning_rate=0.05,
        max_iter=500,
        l2_regularization=1.0,
        random_state=42,
    )
    model.fit(X, y)

    return model, X.columns.tolist()


def fit_ml_dev_model(df_train: pd.DataFrame, term_col: str, fhlb_col: str, lr: LongRunParams):
    """Fit an ML model for Δ(y - y*) instead of Δy.

    This tends to behave better under rate-shock scenarios because y* moves
    monotonically with Rate, and ML only learns the (usually smaller) deviation dynamics.
    """
    tmp = df_train.dropna(subset=["Rate", "Rate_Paid", term_col, fhlb_col]).copy()

    r = tmp["Rate"].to_numpy(dtype=float)
    beta = beta_gompertz(r, lr.k, lr.m, lr.beta_min, lr.beta_max)
    y_star = lr.alpha + beta * r + lr.gamma_ts * tmp[term_col].to_numpy(dtype=float) + lr.gamma_fs * tmp[fhlb_col].to_numpy(dtype=float)
    dev = pd.Series(tmp["Rate_Paid"].to_numpy(dtype=float) - y_star, index=tmp.index)

    dDev = dev.diff()
    dR = tmp["Rate"].diff()
    vol_excess = ewma_vol_excess(dR)

    feats = pd.DataFrame(index=tmp.index)
    feats["DEV_L1"] = dev.shift(1)
    feats["dR"] = dR
    feats["dR_L1"] = dR.shift(1)
    feats["Rate"] = tmp["Rate"]
    feats["TS"] = tmp[term_col]
    feats["FS"] = tmp[fhlb_col]
    feats["dTS"] = tmp[term_col].diff()
    feats["dFS"] = tmp[fhlb_col].diff()
    feats["vol_excess"] = vol_excess
    feats["dDEV_L1"] = dDev.shift(1)

    feats["beta"] = beta
    feats["x_eff"] = beta * tmp["Rate"].to_numpy(dtype=float)
    feats["dx_eff"] = feats["x_eff"].diff()

    data = pd.concat([dDev.rename("dDev"), feats], axis=1).dropna().copy()
    X = data.drop(columns=["dDev"])
    y = data["dDev"].to_numpy(dtype=float)

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        max_depth=3,
        learning_rate=0.05,
        max_iter=500,
        l2_regularization=1.0,
        random_state=42,
    )
    model.fit(X, y)
    return model, X.columns.tolist()


def forecast_ml_delta(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    term_col: str,
    fhlb_col: str,
    lr: LongRunParams,
    model,
    feature_cols: list[str],
    recursive: bool = False,
) -> pd.Series:
    full = pd.concat([df_train, df_test]).copy()
    for c in ["Rate", term_col, fhlb_col, "Rate_Paid"]:
        if c in full.columns:
            full[c] = coerce_numeric(full[c])
    # Allow Rate_Paid to be missing in df_test (scenario forecasting).
    full = full.dropna(subset=["Rate", term_col, fhlb_col]).copy()

    # Precompute long-run y* on full panel
    r = full["Rate"].to_numpy(dtype=float)
    beta = beta_gompertz(r, lr.k, lr.m, lr.beta_min, lr.beta_max)
    y_star = lr.alpha + beta * r + lr.gamma_ts * full[term_col].to_numpy(dtype=float) + lr.gamma_fs * full[fhlb_col].to_numpy(dtype=float)
    y_star = pd.Series(y_star, index=full.index)

    dR = full["Rate"].diff()
    vol_excess = ewma_vol_excess(dR)

    yhat = pd.Series(index=full.index, dtype=float)
    if recursive:
        last_train_date = df_train.index.max()
        yhat.loc[:last_train_date] = full.loc[:last_train_date, "Rate_Paid"]

    test_dates = df_test.index
    for t in test_dates:
        if t not in full.index:
            continue
        idx = full.index.get_loc(t)
        t_prev = full.index[idx - 1]

        # Build features at t
        dr = finite_or_zero(dR.loc[t])
        y_prev = float(yhat.loc[t_prev]) if recursive else float(full.loc[t_prev, "Rate_Paid"])
        row = {
            "ECT_L1": float(y_prev - y_star.loc[t_prev]),
            "dR": dr,
            "dR_L1": finite_or_zero(dR.shift(1).loc[t]),
            "Rate": finite_or_zero(full.loc[t, "Rate"]),
            "TS": finite_or_zero(full.loc[t, term_col]),
            "FS": finite_or_zero(full.loc[t, fhlb_col]),
            "dTS": finite_or_zero(full[term_col].diff().loc[t]),
            "dFS": finite_or_zero(full[fhlb_col].diff().loc[t]),
            "vol_excess": finite_or_zero(vol_excess.loc[t]),
            "beta": float(beta[idx]),
            "x_eff": float(beta[idx] * full.loc[t, "Rate"]),
            "dx_eff": float((beta[idx] * full.loc[t, "Rate"]) - (beta[idx - 1] * full.loc[t_prev, "Rate"])) if idx > 0 else 0.0,
        }

        # lagged dY (actual for one-step)
        if idx >= 2:
            if recursive:
                row["dY_L1"] = float(yhat.loc[t_prev] - yhat.loc[full.index[idx - 2]])
            else:
                row["dY_L1"] = float(full.loc[t_prev, "Rate_Paid"] - full.loc[full.index[idx - 2], "Rate_Paid"])
        else:
            row["dY_L1"] = 0.0

        Xrow = pd.DataFrame([row]).reindex(columns=feature_cols, fill_value=0.0)
        dy_hat = float(model.predict(Xrow)[0])
        if recursive:
            yhat.loc[t] = float(yhat.loc[t_prev] + dy_hat)
        else:
            yhat.loc[t] = float(full.loc[t_prev, "Rate_Paid"] + dy_hat)

    return yhat.loc[df_test.index]


def forecast_ml_dev(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    term_col: str,
    fhlb_col: str,
    lr: LongRunParams,
    model,
    feature_cols: list[str],
    recursive: bool = False,
) -> pd.Series:
    """Forecast y using ML on deviation dynamics: d(y - y*)."""
    full = pd.concat([df_train, df_test]).copy()
    for c in ["Rate", term_col, fhlb_col, "Rate_Paid"]:
        if c in full.columns:
            full[c] = coerce_numeric(full[c])
    full = full.dropna(subset=["Rate", term_col, fhlb_col]).copy()

    r = full["Rate"].to_numpy(dtype=float)
    beta = beta_gompertz(r, lr.k, lr.m, lr.beta_min, lr.beta_max)
    y_star = lr.alpha + beta * r + lr.gamma_ts * full[term_col].to_numpy(dtype=float) + lr.gamma_fs * full[fhlb_col].to_numpy(dtype=float)
    y_star = pd.Series(y_star, index=full.index)

    dR = full["Rate"].diff()
    vol_excess = ewma_vol_excess(dR)

    yhat = pd.Series(index=full.index, dtype=float)
    dev_hat = pd.Series(index=full.index, dtype=float)

    if recursive:
        last_train_date = df_train.index.max()
        yhat.loc[:last_train_date] = full.loc[:last_train_date, "Rate_Paid"]
        dev_hat.loc[:last_train_date] = yhat.loc[:last_train_date] - y_star.loc[:last_train_date]

    test_dates = df_test.index
    for t in test_dates:
        if t not in full.index:
            continue
        idx = full.index.get_loc(t)
        t_prev = full.index[idx - 1]

        if recursive:
            y_prev = float(yhat.loc[t_prev])
            dev_prev = float(dev_hat.loc[t_prev])
            ddev_l1 = float(dev_hat.loc[t_prev] - dev_hat.loc[full.index[idx - 2]]) if idx >= 2 else 0.0
        else:
            y_prev = float(full.loc[t_prev, "Rate_Paid"])
            dev_prev = float(y_prev - y_star.loc[t_prev])
            if idx >= 2:
                y_prev2 = float(full.loc[full.index[idx - 2], "Rate_Paid"])
                dev_prev2 = float(y_prev2 - y_star.loc[full.index[idx - 2]])
                ddev_l1 = float(dev_prev - dev_prev2)
            else:
                ddev_l1 = 0.0

        row = {
            "DEV_L1": finite_or_zero(dev_prev),
            "dR": finite_or_zero(dR.loc[t]),
            "dR_L1": finite_or_zero(dR.shift(1).loc[t]),
            "Rate": finite_or_zero(full.loc[t, "Rate"]),
            "TS": finite_or_zero(full.loc[t, term_col]),
            "FS": finite_or_zero(full.loc[t, fhlb_col]),
            "dTS": finite_or_zero(full[term_col].diff().loc[t]),
            "dFS": finite_or_zero(full[fhlb_col].diff().loc[t]),
            "vol_excess": finite_or_zero(vol_excess.loc[t]),
            "dDEV_L1": finite_or_zero(ddev_l1),
            "beta": float(beta[idx]),
            "x_eff": float(beta[idx] * full.loc[t, "Rate"]),
            "dx_eff": float((beta[idx] * full.loc[t, "Rate"]) - (beta[idx - 1] * full.loc[t_prev, "Rate"])) if idx > 0 else 0.0,
        }

        Xrow = pd.DataFrame([row]).reindex(columns=feature_cols, fill_value=0.0)
        ddev_hat = float(model.predict(Xrow)[0])
        dev_next = float(dev_prev + ddev_hat)
        dev_hat.loc[t] = dev_next
        yhat.loc[t] = float(y_star.loc[t] + dev_next)

    return yhat.loc[df_test.index]


def rmse(a: pd.Series, b: pd.Series) -> float:
    x = (a - b).dropna().to_numpy(dtype=float)
    return float(np.sqrt(np.mean(x**2)))


def main() -> None:
    print("=" * 120)
    print("Two-regime nonlinear-beta ECM + challengers")
    print("=" * 120)

    raw = pd.read_csv("bankratemma.csv")
    raw.rename(columns={"EOM_Dt": "Date", "FEDL01": "Rate", "ILMDHYLD": "Rate_Paid"}, inplace=True)
    raw["Date"] = pd.to_datetime(raw["Date"])
    raw = raw.set_index("Date").sort_index()
    raw = raw.loc[(raw.index >= START_MIN) & (raw.index <= END_MAX)].copy()

    for c in ["Rate", "Rate_Paid"] + TERM_SPREAD_CANDIDATES + FHLB_SPREAD_CANDIDATES:
        if c in raw.columns:
            raw[c] = coerce_numeric(raw[c])
    raw = raw.dropna(subset=["Rate", "Rate_Paid"]).copy()

    summary_rows = []

    for regime_name, cfg in REGIMES.items():
        print("\n" + "=" * 120)
        print(f"{regime_name}")
        print("=" * 120)

        best = choose_window(
            raw,
            start_candidates=cfg["start_candidates"],
            end_candidates=cfg["end_candidates"],
            min_obs=int(cfg["min_obs"]),
        )

        print("\nSelected window:")
        print(
            f"  {best.start}..{best.end} | TS={best.term_col} FS={best.fhlb_col} | train_n={best.n} | RMSE={best.rmse:.4f} | "
            f"ADFn={best.adf_n_p:.4f} KPSS={best.kpss_c_p:.4f} | gate={'PASS' if best.passes_gate else 'FAIL'} | {best.flags}"
        )

        df_win = raw.loc[(raw.index >= best.start) & (raw.index <= best.end)].copy()
        df_win = df_win.dropna(subset=["Rate", "Rate_Paid", best.term_col, best.fhlb_col]).copy()

        df_train = df_win.iloc[: max(0, len(df_win) - TEST_MONTHS)].copy()
        df_test = df_win.iloc[max(0, len(df_win) - TEST_MONTHS) :].copy()

        lr_fit = fit_long_run(df_train, best.term_col, best.fhlb_col)
        if lr_fit is None:
            print("Long-run fit failed on train.")
            continue

        lr_params, resid, _, _, _ = lr_fit
        adf_n_p = safe_adf_p(resid, regression="n")
        kpss_p = safe_kpss_p(resid, regression="c")

        print("\nLong-run (train) residual stationarity screen:")
        print(f"  ADF(n) p={adf_n_p:.4f} | KPSS(c) p={kpss_p:.4f} | gate={'PASS' if (adf_n_p<0.05 and kpss_p>0.05) else 'FAIL'}")

        ecm_sym, cols_sym = fit_ecm(df_train, best.term_col, best.fhlb_col, lr_params, asymmetric=False)
        ecm_asym, cols_asym = fit_ecm(df_train, best.term_col, best.fhlb_col, lr_params, asymmetric=True)

        yhat_sym = forecast_ecm(df_train, df_test, best.term_col, best.fhlb_col, lr_params, ecm_sym, cols_sym, asymmetric=False)
        yhat_asym = forecast_ecm(df_train, df_test, best.term_col, best.fhlb_col, lr_params, ecm_asym, cols_asym, asymmetric=True)

        y_true = df_test["Rate_Paid"]
        rmse_sym = rmse(y_true, yhat_sym)
        rmse_asym = rmse(y_true, yhat_asym)

        ml_model, feat_cols = fit_ml_delta_model(df_train, best.term_col, best.fhlb_col, lr_params)
        yhat_ml = forecast_ml_delta(df_train, df_test, best.term_col, best.fhlb_col, lr_params, ml_model, feat_cols)
        rmse_ml = rmse(y_true, yhat_ml)

        print("\nHoldout RMSE (last 12 months of this window):")
        print(f"  ECM symmetric : {rmse_sym:.4f}")
        print(f"  ECM asymmetric: {rmse_asym:.4f}")
        print(f"  ML challenger : {rmse_ml:.4f}")

        comp = pd.DataFrame(
            {
                "y_true": y_true,
                "yhat_ecm_sym": yhat_sym,
                "yhat_ecm_asym": yhat_asym,
                "yhat_ml": yhat_ml,
            }
        ).dropna()
        if not comp.empty:
            print("\nHoldout last 5 observations (levels):")
            print(comp.tail(5).to_string())

        def diag_u(u: np.ndarray):
            u = pd.Series(u).dropna().to_numpy(dtype=float)
            lb = acorr_ljungbox(u, lags=[6, 12], return_df=True)
            _, arch_p, *_ = het_arch(u, nlags=6)
            _, jb_p, *_ = jarque_bera(u)
            return float(lb.loc[12, "lb_pvalue"]), float(arch_p), float(jb_p)

        lb12s, archps, jbps = diag_u(ecm_sym.resid)
        lb12a, archpa, jbpa = diag_u(ecm_asym.resid)

        summary_rows.append(
            {
                "regime": regime_name,
                "window": f"{best.start}..{best.end}",
                "TS": best.term_col,
                "FS": best.fhlb_col,
                "cointegration_gate_train": bool((adf_n_p < 0.05) and (kpss_p > 0.05)),
                "rmse_holdout_ecm_sym": rmse_sym,
                "rmse_holdout_ecm_asym": rmse_asym,
                "rmse_holdout_ml": rmse_ml,
                "ecm_sym_lb12_p": lb12s,
                "ecm_sym_arch_p": archps,
                "ecm_sym_jb_p": jbps,
                "ecm_asym_lb12_p": lb12a,
                "ecm_asym_arch_p": archpa,
                "ecm_asym_jb_p": jbpa,
            }
        )

    print("\n" + "=" * 120)
    print("Summary table")
    print("=" * 120)

    if summary_rows:
        out_df = pd.DataFrame(summary_rows)
        pd.set_option("display.max_columns", 200)
        print(out_df.to_string(index=False))
    else:
        print("No regimes successfully evaluated.")


if __name__ == "__main__":
    main()
