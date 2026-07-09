"""Robustness checks for the hydro-financial screening.

Re-runs the lagged screening under several robustness conditions and writes a compact summary.
It reproduces the main methods exactly (the same transforms, harmonic de-seasonalization, robust
z-score, market model, forward valuation stress, peak-lag Spearman correlation with circular-shift
surrogates and Benjamini-Hochberg FDR), then varies the choices most likely to affect the retained
signals:

  A. Full 50-pair screen with 5,000 surrogates and BH-FDR.
  B. Non-overlapping 20-day response windows (the forward stress windows overlap heavily).
  C. Moving-block bootstrap 95% confidence interval on each peak correlation.
  D. Lag-grid sensitivity: correlation at neighbouring lags.
  E. Anomaly-definition sensitivity: no linear trend; a single annual harmonic; a standard z-score.
  F. Exclude the 5% of days with the largest absolute market return.

Usage
-----
    python robustness_checks.py --matrix analysis_matrix.csv --out robustness_summary.csv

The matrix is the analysis_matrix.csv written by longtan_hydrofinancial_screening.py. If it does
not already contain the abnormal-return responses or the *_anom anomaly columns, they are rebuilt
from the raw log returns and hydroclimate variables.
"""

import argparse
import json

import numpy as np
import pandas as pd

SEED = 42
LAGS = [0, 1, 3, 5, 10, 20, 30, 45, 60, 90]
N_SURR = 5000
MIN_SHIFT = 60
STRESS_H = 20
BLOCK = 60
N_BOOT = 2000

PHYS = ["Precipitation_mm", "Runoff_Total_mm", "Surface_Runoff_mm", "Baseflow_mm",
        "Root_Moisture_kg_m2", "Evapotranspiration_mm", "Temperature_C",
        "Water_Balance_mm", "Water_Availability_Index", "Dry_Heat_Stress"]
RESP = ["Abnormal_Return", "CAR_5d", "CAR_20d", "Abs_Abnormal_Return", "Forward_AbsAR_20d"]

# Retained signals reported in detail; the last pair is a negative control (precipitation).
FOCUS = [("Root_Moisture_kg_m2", "Forward_AbsAR_20d"),
         ("Baseflow_mm", "Forward_AbsAR_20d"),
         ("Root_Moisture_kg_m2", "Abs_Abnormal_Return"),
         ("Precipitation_mm", "Forward_AbsAR_20d")]


def robust_z(s):
    s = s.astype(float).replace([np.inf, -np.inf], np.nan)
    med = s.median(skipna=True)
    mad = (s - med).abs().median(skipna=True)
    scale = 1.4826 * mad if np.isfinite(mad) and mad > 0 else s.std(skipna=True)
    if not np.isfinite(scale) or scale == 0:
        scale = 1.0
    return (s - med) / scale


def std_z(s):
    s = s.astype(float).replace([np.inf, -np.inf], np.nan)
    sd = s.std(skipna=True) or 1.0
    return (s - s.mean(skipna=True)) / sd


def harmonic_residual(series, include_trend=True, harmonics=(1, 2)):
    s = series.astype(float).replace([np.inf, -np.inf], np.nan)
    out = pd.Series(np.nan, index=s.index, name=s.name)
    mask = s.notna()
    if mask.sum() < 30:
        return out
    dates = s.index[mask]
    t = (dates - dates.min()).days.to_numpy(float)
    ts = (t - t.mean()) / max(t.std(), 1.0)
    ang = 2 * np.pi * dates.dayofyear.to_numpy(float) / 365.25
    cols = [np.ones(mask.sum())]
    if include_trend:
        cols.append(ts)
    for k in harmonics:
        cols += [np.sin(k * ang), np.cos(k * ang)]
    X = np.column_stack(cols)
    y = s.loc[mask].to_numpy(float)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    out.loc[mask] = y - X @ beta
    return out


def transform_phys(series, name):
    s = series.astype(float)
    if name in ["Precipitation_mm", "Runoff_Total_mm", "Surface_Runoff_mm", "Baseflow_mm"]:
        return np.log1p(s.clip(lower=0))
    if name in ["Water_Balance_mm", "Water_Availability_Index", "Dry_Heat_Stress"]:
        return np.arcsinh(s)
    return s


def make_anom(df, col, trend=True, harmonics=(1, 2), zfun=robust_z):
    return zfun(harmonic_residual(transform_phys(df[col], col), include_trend=trend, harmonics=harmonics))


def forward_sum(series, h):
    return series.shift(-1).iloc[::-1].rolling(h, min_periods=h).sum().iloc[::-1]


def market_resid(stock_ret, mkt_ret):
    m = stock_ret.notna() & mkt_ret.notna()
    x = mkt_ret[m].to_numpy(float)
    y = stock_ret[m].to_numpy(float)
    beta = float(np.cov(x, y, ddof=0)[0, 1] / np.var(x))
    alpha = float(y.mean() - beta * x.mean())
    return (stock_ret - (alpha + beta * mkt_ret)).rename("Abnormal_Return"), alpha, beta


def rankarr(s):
    return pd.Series(s).replace([np.inf, -np.inf], np.nan).rank(method="average").to_numpy(float)


def pearson(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 8:
        return np.nan
    xv = x[m] - x[m].mean()
    yv = y[m] - y[m].mean()
    den = np.sqrt((xv * xv).sum() * (yv * yv).sum())
    return float((xv * yv).sum() / den) if den > 0 else np.nan


def lagcorr(xr, yr, lag):
    xr2, yr2 = (xr, yr) if lag == 0 else (xr[:-lag], yr[lag:])
    return pearson(xr2, yr2)


def peak_lag(xr, yr, lags=LAGS):
    vals = {L: lagcorr(xr, yr, L) for L in lags}
    vals = {L: v for L, v in vals.items() if np.isfinite(v)}
    bl = max(vals, key=lambda L: abs(vals[L]))
    return bl, vals[bl], vals


def surrogate_p(xr, yr, n_surr=N_SURR, lags=LAGS, min_shift=MIN_SHIFT, seed=SEED):
    rng = np.random.default_rng(seed)
    _, obs, _ = peak_lag(xr, yr, lags)
    n = len(yr)
    shifts = np.arange(min_shift, n - min_shift)
    null = np.empty(n_surr)
    for b in range(n_surr):
        yp = np.roll(yr, int(rng.choice(shifts)))
        null[b] = np.nanmax([abs(lagcorr(xr, yp, L)) for L in lags])
    return (1 + np.sum(null >= abs(obs))) / (1 + n_surr), obs


def fdr_bh(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = p[order] * m / (np.arange(m) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.clip(adj, 0, 1)
    return out


def build_from_raw(df):
    """Rebuild abnormal-return responses from a raw synchronized matrix."""
    df = df.copy()
    if "Runoff_Total_mm" not in df and {"Surface_Runoff_mm", "Baseflow_mm"} <= set(df):
        df["Runoff_Total_mm"] = df["Surface_Runoff_mm"] + df["Baseflow_mm"]
    ar, alpha, beta = market_resid(df["Log_Return"], df["Market_Log_Return"])
    df["Abnormal_Return"] = ar
    df["Abs_Abnormal_Return"] = ar.abs()
    df["CAR_5d"] = forward_sum(ar, 5)
    df["CAR_20d"] = forward_sum(ar, 20)
    df["Forward_AbsAR_20d"] = forward_sum(ar.abs(), STRESS_H)
    return df, alpha, beta


def block_bootstrap_ci(base, pc, rc, peak, rng):
    """Moving-block bootstrap 95% CI on the peak-lag rank correlation.

    The lag-aligned pairs (x_t, y_{t+peak}) are formed BEFORE block-resampling, and the
    lag-0 rank correlation is taken on each resample. Resampling contemporaneous pairs and
    re-applying the lag afterwards is invalid when the block length is shorter than the lag:
    it forces the lagged pairs across independent blocks and collapses the bootstrap
    distribution toward zero, so the point estimate can fall outside its own interval.
    Pre-alignment avoids this and the bootstrap mean sits on the observed correlation.
    """
    xa = base[f"{pc}_anom"].to_numpy(float)
    ya = base[rc].to_numpy(float)
    if peak > 0:
        xa, ya = xa[:-peak], ya[peak:]
    mfin = np.isfinite(xa) & np.isfinite(ya)
    xa, ya = xa[mfin], ya[mfin]
    na = len(xa)
    nbk = int(np.ceil(na / BLOCK))
    boots = np.empty(N_BOOT)
    for b in range(N_BOOT):
        starts = rng.integers(0, na - BLOCK, nbk)
        idx = np.concatenate([np.arange(s, s + BLOCK) for s in starts])[:na]
        boots[b] = pearson(rankarr(xa[idx]), rankarr(ya[idx]))
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return float(np.nanmean(boots)), float(lo), float(hi)


def main(matrix_path, out_path):
    df = pd.read_csv(matrix_path, parse_dates=["Date"]).sort_values("Date").set_index("Date")
    df = df.replace([np.inf, -np.inf], np.nan)
    if "Forward_AbsAR_20d" not in df.columns:
        df, alpha, beta = build_from_raw(df)
        print(f"Rebuilt responses from raw matrix (alpha={alpha:.6f}, beta={beta:.3f}).")
    for c in PHYS:
        if f"{c}_anom" not in df.columns:
            df[f"{c}_anom"] = make_anom(df, c)
    n = len(df)
    print(f"Matrix: {df.index.min().date()}..{df.index.max().date()}  n={n} trading days")

    # A. Full 50-pair screen with 5,000 surrogates and BH-FDR.
    print(f"\n[A] {len(PHYS) * len(RESP)}-pair screen with {N_SURR} surrogates + BH-FDR ...")
    keys, praw, prho, plag = [], [], [], []
    for pc in PHYS:
        xr = rankarr(df[f"{pc}_anom"])
        for rc in RESP:
            p, _ = surrogate_p(xr, rankarr(df[rc]))
            bl, rho, _ = peak_lag(xr, rankarr(df[rc]))
            keys.append((pc, rc)); praw.append(p); prho.append(rho); plag.append(bl)
    q = fdr_bh(praw)
    qmap = {keys[i]: (prho[i], plag[i], praw[i], q[i]) for i in range(len(keys))}
    n_sig = int(np.sum(q < 0.05))
    print(f"    pairs with q<0.05: {n_sig}")

    # B-F. Focus signals.
    rng = np.random.default_rng(SEED)
    rows = []
    for pc, rc in FOCUS:
        xr = rankarr(df[f"{pc}_anom"]); yr = rankarr(df[rc])
        bl, rho, curve = peak_lag(xr, yr)
        rho_base, q_base = qmap[(pc, rc)][0], qmap[(pc, rc)][3]

        # B. Non-overlapping windows: sample every STRESS_H-th day and correlate at the
        #    expected lead expressed in non-overlapping steps.
        idx = np.arange(0, n, STRESS_H)
        xr_no = rankarr(df[f"{pc}_anom"].iloc[idx]); yr_no = rankarr(df[rc].iloc[idx])
        step = max(1, round(bl / STRESS_H))
        rho_no = lagcorr(xr_no, yr_no, step)

        # C. Moving-block bootstrap CI.
        boot_mean, lo, hi = block_bootstrap_ci(df, pc, rc, bl, rng)

        # E. Anomaly-definition sensitivity.
        alt_no_trend = peak_lag(rankarr(make_anom(df, pc, trend=False)), yr)[1]
        alt_one_harm = peak_lag(rankarr(make_anom(df, pc, harmonics=(1,))), yr)[1]
        alt_std_z = peak_lag(rankarr(make_anom(df, pc, zfun=std_z)), yr)[1]

        # F. Exclude the top 5% of days by absolute market return.
        thr = df["Market_Log_Return"].abs().quantile(0.95)
        keep = df["Market_Log_Return"].abs() <= thr
        rho_excl = peak_lag(rankarr(df.loc[keep, f"{pc}_anom"]), rankarr(df.loc[keep, rc]))[1]

        rows.append(dict(
            physical=pc, response=rc, peak_lag=bl, rho_baseline=round(rho_base, 3),
            q_baseline=round(q_base, 4), rho_nonoverlap=round(rho_no, 3),
            rho_block_mean=round(boot_mean, 3), rho_block_CI=f"[{lo:.2f}, {hi:.2f}]",
            rho_no_trend=round(alt_no_trend, 3), rho_one_harmonic=round(alt_one_harm, 3),
            rho_standard_z=round(alt_std_z, 3), rho_excl_top5pct_mktdays=round(rho_excl, 3),
            lag_curve={int(k): (round(v, 3) if np.isfinite(v) else None) for k, v in curve.items()},
        ))
        print(f"  {pc} -> {rc}: peak lag {bl}, rho {rho:.3f} (q={q_base:.4f}); "
              f"non-overlap {rho_no:.3f}; block mean {boot_mean:.3f} CI [{lo:.2f},{hi:.2f}]; "
              f"alt-anom {alt_no_trend:.3f}/{alt_one_harm:.3f}/{alt_std_z:.3f}; excl-top5% {rho_excl:.3f}")

    pd.DataFrame(rows).to_csv(out_path, index=False)
    with open(out_path.replace(".csv", ".json"), "w", encoding="utf-8") as f:
        json.dump({"n_trading_days": n, "n_pairs": len(PHYS) * len(RESP),
                   "n_surrogates": N_SURR, "n_pairs_q_lt_0.05": n_sig, "focus": rows}, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--matrix", required=True, help="Path to analysis_matrix.csv.")
    parser.add_argument("--out", default="robustness_summary.csv", help="Output CSV path.")
    args = parser.parse_args()
    main(args.matrix, args.out)
