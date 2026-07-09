"""Hydro-financial screening of climate variability and hydropower-utility valuation.

Reproducible analysis code for:

    G. Al Zaabi, Y. Wehbe, Y. Zhu and H. Fu, "Slow basin storage anomalies precede
    valuation stress for the Longtan hydropower operator."

The pipeline has three stages:

  1. Physical. Delineate the catchment upstream of the Longtan Hydropower Station from
     HydroBASINS, extract daily basin-mean hydroclimate variables (precipitation,
     temperature, surface runoff, baseflow, root-zone soil moisture, evapotranspiration)
     from public Earth-observation datasets in Google Earth Engine, derive water-balance,
     water-availability and dry-heat-stress indicators, and convert every variable into a
     de-seasonalized, robustly standardized anomaly.

  2. Financial. Convert the daily adjusted close of the listed operator (GGEPC, ticker
     600236) and of the market index into log returns, remove broad market movement with a
     one-factor market model, and build forward abnormal-return and forward 20-day
     valuation-stress responses.

  3. Statistical. Screen lagged Spearman associations between each hydroclimate anomaly and
     each valuation response, assess significance with a circular-shift surrogate test that
     preserves autocorrelation, and control multiple testing with the Benjamini-Hochberg
     false-discovery-rate procedure.

Data sources
------------
Hydroclimate (openly available in Google Earth Engine):
    CHIRPS Daily .................. UCSB-CHG/CHIRPS/DAILY
    ERA5-Land Daily Aggregated .... ECMWF/ERA5_LAND/DAILY_AGGR
    GLDAS Noah v2.1 ............... NASA/GLDAS/V021/NOAH/G025/T3H
    HydroSHEDS / HydroBASINS ...... WWF/HydroSHEDS/...
Equity and market index (licensed, NOT redistributed):
    Daily adjusted close for GGEPC (600236) and the Shanghai Composite Index (000001) from
    the China Stock Market and Accounting Research (CSMAR) database. Supply these locally as
    a CSV (see --finance and the expected schema in load_financial_csv). They are used only
    to compute daily returns; no raw CSMAR field is written to the outputs.

Usage
-----
    # Reuse a previously prepared analysis matrix (fastest; no Earth Engine or price data needed):
    python longtan_hydrofinancial_screening.py --matrix analysis_matrix.csv --out ./outputs

    # Build the matrix from scratch (requires an Earth Engine project and a CSMAR price CSV):
    export GEE_PROJECT_ID=your-earth-engine-project
    python longtan_hydrofinancial_screening.py --finance csmar_prices.csv --out ./outputs

Requires Python 3.10+ and the packages listed in requirements.txt. Earth Engine, geemap,
geopandas and pyproj are needed only for a fresh extraction and are imported lazily.
"""

import argparse
import importlib
import json
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
SEED = 42

TICKER = "600236"                 # GGEPC, Shanghai Stock Exchange
MARKET_TICKER = "000001"          # Shanghai Composite Index
COMPANY_LABEL = "Guangxi Guiguan Electric Power Co., Ltd."
DAM_LON_LAT = [107.03, 25.04]     # approximate Longtan Dam outlet
HYDROBASINS_LEVEL = 7

START_DATE = "2020-01-01"
REQUESTED_END_DATE = "2026-05-26"

# Earth Engine reduction settings (fresh extraction only)
GEE_REDUCTION_SCALE_METERS = 25000
GEE_TILE_SCALE = 4
GEE_DAILY_CHUNK_DAYS = 31
GEE_MAX_CHUNK_RETRIES = 2
GEE_MIN_SPLIT_DAYS = 7
GEE_SIMPLIFY_TOLERANCE_METERS = 5000

# Screening settings (these reproduce the published results)
LAGS_DAYS = [0, 1, 3, 5, 10, 20, 30, 45, 60, 90]
N_SURROGATES = 5000
FDR_ALPHA = 0.10

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 360,
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.frameon": True,
    "grid.alpha": 0.22,
})

PHYSICAL_COLUMNS = [
    "Precipitation_mm", "Runoff_Total_mm", "Surface_Runoff_mm", "Baseflow_mm",
    "Root_Moisture_kg_m2", "Evapotranspiration_mm", "Temperature_C",
    "Water_Balance_mm", "Water_Availability_Index", "Dry_Heat_Stress",
]
PHYSICAL_LABELS = {
    "Precipitation_mm": "Precipitation", "Runoff_Total_mm": "Total runoff",
    "Surface_Runoff_mm": "Surface runoff", "Baseflow_mm": "Baseflow",
    "Root_Moisture_kg_m2": "Root-zone soil moisture", "Evapotranspiration_mm": "Evapotranspiration",
    "Temperature_C": "Temperature", "Water_Balance_mm": "Water balance",
    "Water_Availability_Index": "Water availability", "Dry_Heat_Stress": "Dry-heat stress",
}
VALUATION_RESPONSE_COLUMNS = ["Abnormal_Return", "CAR_5d", "CAR_20d", "Abs_Abnormal_Return", "Forward_AbsAR_20d"]
VALUATION_LABELS = {
    "Abnormal_Return": "Market-adjusted return", "CAR_5d": "Forward 5-day CAR",
    "CAR_20d": "Forward 20-day CAR", "Abs_Abnormal_Return": "Absolute abnormal return",
    "Forward_AbsAR_20d": "Forward 20-day valuation stress",
}
UNITS = {
    "Adjusted_Close": "CNY", "Relative_Valuation_Index": "index", "Abs_Abnormal_Return": "abs log return",
    "Precipitation_mm": "mm/day", "Runoff_Total_mm": "mm/day", "Surface_Runoff_mm": "mm/day",
    "Baseflow_mm": "mm/day", "Root_Moisture_kg_m2": "kg/m2", "Evapotranspiration_mm": "mm/day",
    "Temperature_C": "deg C", "Water_Balance_mm": "mm/day", "Water_Availability_Index": "index",
    "Dry_Heat_Stress": "index",
}
COLORS = {
    "Adjusted_Close": "#174a68", "Relative_Valuation_Index": "#444444", "Abs_Abnormal_Return": "#8f2d56",
    "Precipitation_mm": "#2b83ba", "Runoff_Total_mm": "#1b9e77", "Surface_Runoff_mm": "#43aa8b",
    "Baseflow_mm": "#4d908e", "Root_Moisture_kg_m2": "#577590", "Evapotranspiration_mm": "#90be6d",
    "Temperature_C": "#f9844a", "Water_Balance_mm": "#2a9d8f", "Water_Availability_Index": "#006d77",
    "Dry_Heat_Stress": "#c1121f",
}


# --------------------------------------------------------------------------------------
# Statistical and signal-processing helpers
# --------------------------------------------------------------------------------------
def fdr_bh(p_values):
    """Benjamini-Hochberg adjusted p-values (q-values)."""
    p = np.asarray(p_values, dtype=float)
    q = np.full_like(p, np.nan)
    mask = np.isfinite(p)
    vals = p[mask]
    if len(vals) == 0:
        return q
    order = np.argsort(vals)
    ranked = vals[order]
    m = len(ranked)
    adjusted = ranked * m / (np.arange(m) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.clip(adjusted, 0, 1)
    q[mask] = restored
    return q


def pearson_corr(x, y):
    """Pearson correlation on finite pairs; used on ranks to obtain a Spearman correlation."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 8:
        return np.nan
    xv = x[mask] - np.nanmean(x[mask])
    yv = y[mask] - np.nanmean(y[mask])
    den = np.sqrt(np.sum(xv * xv) * np.sum(yv * yv))
    return float(np.sum(xv * yv) / den) if den > 0 else np.nan


def rank_array(series):
    return series.replace([np.inf, -np.inf], np.nan).rank(method="average").to_numpy(dtype=float)


def robust_zscore(series):
    """Standardize by the median and the median absolute deviation (scaled by 1.4826)."""
    s = series.astype(float).replace([np.inf, -np.inf], np.nan)
    med = s.median(skipna=True)
    mad = (s - med).abs().median(skipna=True)
    scale = 1.4826 * mad if np.isfinite(mad) and mad > 0 else s.std(skipna=True)
    if not np.isfinite(scale) or scale == 0:
        scale = 1.0
    return (s - med) / scale


def harmonic_residual(series, include_trend=True):
    """De-seasonalize by regressing on a linear trend plus annual and semi-annual harmonics."""
    s = series.astype(float).replace([np.inf, -np.inf], np.nan)
    out = pd.Series(np.nan, index=s.index, name=s.name)
    mask = s.notna()
    if mask.sum() < 30:
        return out
    dates = s.index[mask]
    t_days = (dates - dates.min()).days.to_numpy(dtype=float)
    t_scaled = (t_days - t_days.mean()) / max(t_days.std(), 1.0)
    angle = 2.0 * np.pi * dates.dayofyear.to_numpy(dtype=float) / 365.25
    cols = [np.ones(mask.sum())]
    if include_trend:
        cols.append(t_scaled)
    for k in [1, 2]:
        cols.append(np.sin(k * angle))
        cols.append(np.cos(k * angle))
    x = np.column_stack(cols)
    y = s.loc[mask].to_numpy(dtype=float)
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    out.loc[mask] = y - x @ beta
    return out


def transform_physical(series, name):
    """Distribution-aware transform: log1p for non-negative fluxes, asinh for signed indices."""
    s = series.astype(float)
    if name in ["Precipitation_mm", "Runoff_Total_mm", "Surface_Runoff_mm", "Baseflow_mm"]:
        return np.log1p(s.clip(lower=0))
    if name in ["Water_Balance_mm", "Water_Availability_Index", "Dry_Heat_Stress"]:
        return np.arcsinh(s)
    return s


def forward_sum(series, horizon):
    """Forward-looking rolling sum over the next `horizon` observations (excludes the current day)."""
    return series.shift(-1).iloc[::-1].rolling(horizon, min_periods=horizon).sum().iloc[::-1]


def market_model_residual(stock_return, market_return):
    """One-factor market model; returns the abnormal-return residual and the fitted alpha, beta."""
    mask = stock_return.notna() & market_return.notna()
    x = market_return.loc[mask].to_numpy(dtype=float)
    y = stock_return.loc[mask].to_numpy(dtype=float)
    beta = float(np.cov(x, y, ddof=0)[0, 1] / np.var(x))
    alpha = float(np.mean(y) - beta * np.mean(x))
    residual = stock_return - (alpha + beta * market_return)
    return residual.rename("Abnormal_Return"), alpha, beta


def autocorr_values(series, max_lag=120):
    s = series.dropna().astype(float)
    return np.array([1.0 if lag == 0 else s.autocorr(lag) for lag in range(max_lag + 1)], dtype=float)


def effective_sample_size(series, max_lag=120):
    """Effective sample size from the integrated autocorrelation time of a series."""
    s = series.dropna().astype(float)
    n = len(s)
    acf = autocorr_values(s, max_lag=max_lag)
    positive = []
    first_zero = max_lag
    for lag in range(1, len(acf)):
        if not np.isfinite(acf[lag]) or acf[lag] <= 0:
            first_zero = lag
            break
        positive.append(acf[lag])
    tau = 1.0 + 2.0 * float(np.nansum(positive))
    return n / max(tau, 1.0), first_zero, tau


def add_physical_indexes(matrix):
    """Add the pre-specified water-availability and dry-heat-stress indices.

    Weights encode physical direction only (water inputs and storage positive;
    evapotranspiration and temperature negative) and are not fitted to the financial data.
    """
    out = matrix.copy()
    components = {
        "Precipitation_mm": 1.0, "Runoff_Total_mm": 1.0, "Baseflow_mm": 0.5,
        "Root_Moisture_kg_m2": 0.7, "Evapotranspiration_mm": -0.8, "Temperature_C": -0.5,
    }
    signed = {}
    for col, sign in components.items():
        signed[col] = sign * (out[col] - out[col].mean()) / (out[col].std(ddof=0) or 1.0)
    out["Water_Availability_Index"] = pd.concat(signed, axis=1).mean(axis=1)
    out["Dry_Heat_Stress"] = (
        (out["Temperature_C"] - out["Temperature_C"].mean()) / (out["Temperature_C"].std(ddof=0) or 1.0)
        - (out["Precipitation_mm"] - out["Precipitation_mm"].mean()) / (out["Precipitation_mm"].std(ddof=0) or 1.0)
    )
    return out


# --------------------------------------------------------------------------------------
# Google Earth Engine extraction (used only for a fresh build)
# --------------------------------------------------------------------------------------
def _ensure_package(pip_name, import_name=None):
    module_name = import_name or pip_name
    try:
        return importlib.import_module(module_name)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name])
        return importlib.import_module(module_name)


def initialize_earth_engine(project_id):
    ee = _ensure_package("earthengine-api", "ee")
    init_kwargs = {"project": project_id} if project_id else {}
    try:
        ee.Initialize(**init_kwargs)
    except Exception:
        ee.Authenticate()
        ee.Initialize(**init_kwargs)
    return ee


def build_upstream_watershed(ee, dam_lon_lat, level):
    """Delineate the catchment upstream of the dam outlet from the HydroBASINS topology."""
    dam_point = ee.Geometry.Point(dam_lon_lat)
    basins = ee.FeatureCollection(f"WWF/HydroSHEDS/v1/Basins/hybas_{level}")
    target = ee.Feature(basins.filterBounds(dam_point).first())
    target_id = int(target.get("HYBAS_ID").getInfo())
    main_id = int(target.get("MAIN_BAS").getInfo())
    candidates = basins.filter(ee.Filter.eq("MAIN_BAS", main_id))
    attrs = candidates.select(["HYBAS_ID", "NEXT_DOWN", "SUB_AREA", "UP_AREA"]).getInfo()["features"]
    next_down_by_id = {
        int(f["properties"]["HYBAS_ID"]): int(f["properties"].get("NEXT_DOWN") or 0) for f in attrs
    }
    upstream_ids = {target_id}
    changed = True
    while changed:
        changed = False
        for hybas_id, next_down in next_down_by_id.items():
            if hybas_id not in upstream_ids and next_down in upstream_ids:
                upstream_ids.add(hybas_id)
                changed = True
    watershed_fc = candidates.filter(ee.Filter.inList("HYBAS_ID", sorted(upstream_ids)))
    watershed_geom = watershed_fc.geometry().dissolve(maxError=1000)
    area_km2 = watershed_geom.area(maxError=1000).divide(1e6).getInfo()
    metadata = {
        "target_hybas_id": target_id, "main_bas": main_id,
        "n_upstream_basins": len(upstream_ids), "area_km2": area_km2, "hydrobasins_level": level,
    }
    return dam_point, watershed_fc, watershed_geom, metadata


def _latest_image_date(ee, collection_id):
    millis = ee.ImageCollection(collection_id).aggregate_max("system:time_start")
    return pd.Timestamp(ee.Date(millis).format("YYYY-MM-dd").getInfo())


def compute_common_daily_end(ee, requested_end):
    """Latest date for which all source collections have data (data availability differs by product)."""
    requested = pd.Timestamp(requested_end).normalize()
    ends = {
        "CHIRPS": _latest_image_date(ee, "UCSB-CHG/CHIRPS/DAILY"),
        "ERA5_Land": _latest_image_date(ee, "ECMWF/ERA5_LAND/DAILY_AGGR"),
        "GLDAS": _latest_image_date(ee, "NASA/GLDAS/V021/NOAH/G025/T3H"),
    }
    return min([requested] + list(ends.values())).normalize()


def _ee_to_df_with_retries(feature_collection, label):
    geemap = _ensure_package("geemap", "geemap")
    last_error = None
    for attempt in range(1, GEE_MAX_CHUNK_RETRIES + 1):
        try:
            return geemap.ee_to_df(feature_collection)
        except Exception as exc:
            last_error = exc
            if attempt < GEE_MAX_CHUNK_RETRIES:
                time.sleep(8 * attempt)
    raise last_error


def _iter_date_chunks(start_date, end_date, chunk_days):
    chunk_start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    while chunk_start <= end:
        chunk_end = min(chunk_start + pd.Timedelta(days=chunk_days - 1), end)
        yield chunk_start, chunk_end
        chunk_start = chunk_end + pd.Timedelta(days=1)


def _gee_daily_feature_collection(ee, reduction_geom, start_day, end_day):
    start = ee.Date(pd.Timestamp(start_day).strftime("%Y-%m-%d"))
    end = ee.Date(pd.Timestamp(end_day).strftime("%Y-%m-%d")).advance(1, "day")
    offsets = ee.List.sequence(0, end.difference(start, "day").subtract(1))
    chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").select("precipitation")
    era5 = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").select("temperature_2m")
    gldas = ee.ImageCollection("NASA/GLDAS/V021/NOAH/G025/T3H").select(
        ["Qs_acc", "Qsb_acc", "RootMoist_inst", "Evap_tavg"]
    )

    def one_day(offset):
        day_start = start.advance(offset, "day")
        day_end = day_start.advance(1, "day")
        precip = chirps.filterDate(day_start, day_end).sum().rename("Precipitation_mm")
        temp_c = era5.filterDate(day_start, day_end).mean().subtract(273.15).rename("Temperature_C")
        gldas_day = gldas.filterDate(day_start, day_end)
        surface_runoff = gldas_day.select("Qs_acc").sum().rename("Surface_Runoff_mm")
        baseflow = gldas_day.select("Qsb_acc").sum().rename("Baseflow_mm")
        root_moisture = gldas_day.select("RootMoist_inst").mean().rename("Root_Moisture_kg_m2")
        evap = gldas_day.select("Evap_tavg").mean().multiply(86400).rename("Evapotranspiration_mm")
        image = ee.Image.cat([precip, surface_runoff, baseflow, root_moisture, evap, temp_c]).toFloat()
        stats = image.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=reduction_geom,
            scale=GEE_REDUCTION_SCALE_METERS, maxPixels=1e13, tileScale=GEE_TILE_SCALE,
        )
        return ee.Feature(None, stats).set(
            {"Date": day_start.format("YYYY-MM-dd"), "system:time_start": day_start.millis()}
        )

    return ee.FeatureCollection(offsets.map(one_day))


def _extract_daily_chunk(ee, reduction_geom, start_day, end_day):
    """Extract one date range; on failure, bisect until chunks are small enough to succeed."""
    start_day = pd.Timestamp(start_day).normalize()
    end_day = pd.Timestamp(end_day).normalize()
    n_days = (end_day - start_day).days + 1
    label = f"{start_day.date()} to {end_day.date()}"
    keep = ["Date", "Precipitation_mm", "Surface_Runoff_mm", "Baseflow_mm",
            "Root_Moisture_kg_m2", "Evapotranspiration_mm", "Temperature_C"]
    try:
        fc = _gee_daily_feature_collection(ee, reduction_geom, start_day, end_day)
        chunk = _ee_to_df_with_retries(fc, label)
        chunk["Date"] = pd.to_datetime(chunk["Date"])
        return chunk[[c for c in keep if c in chunk.columns]]
    except Exception as exc:
        if n_days <= GEE_MIN_SPLIT_DAYS:
            raise RuntimeError(f"GEE chunk failed at {n_days} days for {label}") from exc
        mid = start_day + pd.Timedelta(days=n_days // 2 - 1)
        left = _extract_daily_chunk(ee, reduction_geom, start_day, mid)
        right = _extract_daily_chunk(ee, reduction_geom, mid + pd.Timedelta(days=1), end_day)
        return pd.concat([left, right], ignore_index=True).drop_duplicates("Date").sort_values("Date")


def extract_climate_from_gee(project_id, fig_dir):
    """Delineate the basin, draw the study-area map, and extract the daily basin-mean climate table."""
    ee = initialize_earth_engine(project_id)
    _, watershed_fc, watershed_geom, watershed_meta = build_upstream_watershed(ee, DAM_LON_LAT, HYDROBASINS_LEVEL)
    reduction_geom = watershed_geom.simplify(maxError=GEE_SIMPLIFY_TOLERANCE_METERS)
    plot_study_area(ee, watershed_fc, watershed_geom, watershed_meta, fig_dir)

    common_end = compute_common_daily_end(ee, REQUESTED_END_DATE)
    frames = [
        _extract_daily_chunk(ee, reduction_geom, start, end)
        for start, end in _iter_date_chunks(START_DATE, common_end, GEE_DAILY_CHUNK_DAYS)
    ]
    climate = pd.concat(frames, ignore_index=True).drop_duplicates("Date").sort_values("Date").set_index("Date")
    climate = climate.apply(pd.to_numeric, errors="coerce").asfreq("D").interpolate(method="time", limit=3)
    climate["Water_Balance_mm"] = climate["Precipitation_mm"] - climate["Evapotranspiration_mm"]
    climate["Runoff_Total_mm"] = climate["Surface_Runoff_mm"] + climate["Baseflow_mm"]
    climate["Runoff_Ratio"] = climate["Runoff_Total_mm"] / (climate["Precipitation_mm"].abs() + 1e-6)
    return climate


def plot_study_area(ee, watershed_fc, watershed_geom, watershed_meta, fig_dir):
    """Study-area map: upstream basin, river network and dam location (Figure 1 basemap)."""
    geemap = _ensure_package("geemap", "geemap")
    gpd = _ensure_package("geopandas", "geopandas")
    pyproj = _ensure_package("pyproj", "pyproj")

    basin_gdf = geemap.ee_to_gdf(watershed_fc).set_crs("EPSG:4326", allow_override=True)
    basin_union = basin_gdf.dissolve()
    rivers_fc = ee.FeatureCollection("WWF/HydroSHEDS/v1/FreeFlowingRivers").filterBounds(watershed_geom)
    rivers_gdf = geemap.ee_to_gdf(rivers_fc)
    if not rivers_gdf.empty:
        rivers_gdf = rivers_gdf.set_crs("EPSG:4326", allow_override=True)
    dam_gdf = gpd.GeoDataFrame(
        {"name": ["Longtan Dam"]},
        geometry=gpd.points_from_xy([DAM_LON_LAT[0]], [DAM_LON_LAT[1]]), crs="EPSG:4326",
    )

    fig, ax = plt.subplots(figsize=(9.5, 8.2))
    basin_union.plot(ax=ax, facecolor="#e7f0df", edgecolor="#245c3a", linewidth=1.8)
    basin_gdf.boundary.plot(ax=ax, color="#74a57f", linewidth=0.45, alpha=0.75)
    if not rivers_gdf.empty:
        rivers_gdf.plot(ax=ax, color="#2b8cbe", linewidth=0.8, alpha=0.75)
    dam_gdf.plot(ax=ax, color="#d00000", edgecolor="white", markersize=90, zorder=4)
    ax.text(DAM_LON_LAT[0] + 0.08, 24.75, "Longtan Dam", fontsize=10, weight="bold", color="#7a0000")
    ax.annotate("N", xy=(0.92, 0.28), xytext=(0.92, 0.18), xycoords="axes fraction",
                ha="center", va="center", fontsize=13, fontweight="bold",
                arrowprops=dict(facecolor="black", width=3, headwidth=12, shrink=0))
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    lat = 23.0
    lon0 = xlim[0] + 0.08 * (xlim[1] - xlim[0])
    lon1, lat1, _ = pyproj.Geod(ellps="WGS84").fwd(lon0, lat, 90, 100000)
    ax.plot([lon0, lon1], [lat, lat1], color="black", lw=3, solid_capstyle="butt")
    ax.text((lon0 + lon1) / 2, lat + 0.02 * (ylim[1] - ylim[0]), "100 km", ha="center", va="bottom", fontsize=10)
    minx, miny, maxx, maxy = basin_union.total_bounds
    ax.set_xlim(minx - 0.08 * (maxx - minx), maxx + 0.08 * (maxx - minx))
    ax.set_ylim(miny - 0.08 * (maxy - miny), maxy + 0.08 * (maxy - miny))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Upstream basin of the Longtan Hydropower Station\n"
                 f"HydroBASINS level {HYDROBASINS_LEVEL}; area {watershed_meta['area_km2']:,.0f} km2")
    ax.grid(True)
    _savefig(fig, fig_dir / "fig1_study_area_basin.png")


# --------------------------------------------------------------------------------------
# Financial data (CSMAR)
# --------------------------------------------------------------------------------------
def load_financial_csv(path):
    """Load daily adjusted close for the company and the market index from a CSMAR export.

    Expected columns (case-sensitive):
        Date                    parseable date
        Adjusted_Close          split/dividend-adjusted close for GGEPC (ticker 600236)
        Market_Adjusted_Close   close (or adjusted close) for the Shanghai Composite Index (000001)

    Returns a daily-indexed frame with log returns for both series.
    """
    fin = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date").set_index("Date")
    required = {"Adjusted_Close", "Market_Adjusted_Close"}
    missing = required - set(fin.columns)
    if missing:
        raise ValueError(f"Financial CSV is missing required columns: {sorted(missing)}")
    fin = fin[["Adjusted_Close", "Market_Adjusted_Close"]].dropna()
    fin["Log_Return"] = np.log(fin["Adjusted_Close"]).diff()
    fin["Market_Log_Return"] = np.log(fin["Market_Adjusted_Close"]).diff()
    return fin


def assemble_matrix(climate, finance):
    """Synchronize the daily climate and financial records onto the trading calendar."""
    matrix = finance.join(climate, how="left").sort_index()
    fill_cols = [c for c in climate.columns if c in matrix.columns]
    matrix[fill_cols] = matrix[fill_cols].interpolate(method="time", limit=3).ffill(limit=2)
    matrix = matrix.dropna()
    return add_physical_indexes(matrix)


# --------------------------------------------------------------------------------------
# Anomaly and valuation-metric construction
# --------------------------------------------------------------------------------------
def build_features(df):
    """Add abnormal-return responses and de-seasonalized hydroclimate anomalies in place."""
    df["Abnormal_Return"], alpha, beta = market_model_residual(df["Log_Return"], df["Market_Log_Return"])
    df["Abs_Abnormal_Return"] = df["Abnormal_Return"].abs()
    df["CAR_5d"] = forward_sum(df["Abnormal_Return"], 5)
    df["CAR_20d"] = forward_sum(df["Abnormal_Return"], 20)
    df["Forward_AbsAR_20d"] = forward_sum(df["Abs_Abnormal_Return"], 20)
    df["Relative_Valuation_Index"] = np.exp(df["Abnormal_Return"].fillna(0).cumsum())
    for col in PHYSICAL_COLUMNS:
        df[f"{col}_anom"] = robust_zscore(harmonic_residual(transform_physical(df[col], col), include_trend=True))
    return df, alpha, beta


def surrogate_min_shift(df):
    """Minimum circular-shift offset, from the median lag at which autocorrelation first turns non-positive."""
    first_zero = []
    for col in PHYSICAL_COLUMNS + ["Abnormal_Return", "Forward_AbsAR_20d"]:
        _, fz, _ = effective_sample_size(df[col], max_lag=120)
        first_zero.append(fz)
    med = pd.Series(first_zero).replace(0, np.nan).median()
    return int(max(20, min(90, round(med))))


# --------------------------------------------------------------------------------------
# Lagged screening, circular-shift surrogate test, false-discovery control
# --------------------------------------------------------------------------------------
def _lag_corr(x_rank, y_rank, lag):
    if lag == 0:
        xr, yr = x_rank, y_rank
    else:
        xr, yr = x_rank[:-lag], y_rank[lag:]
    mask = np.isfinite(xr) & np.isfinite(yr)
    if mask.sum() < 8:
        return np.nan
    return pearson_corr(xr[mask], yr[mask])


def max_lag_spearman_test(x_rank, y_rank, min_shift, rng):
    """Peak-lag Spearman correlation with a circular-shift surrogate p-value.

    The peak statistic is the maximum absolute correlation over the lag grid. Each surrogate
    circularly shifts the response, repeats the same lag search, and records its own peak
    absolute correlation, so the lag selection is absorbed into the null.
    """
    observed = {lag: _lag_corr(x_rank, y_rank, lag) for lag in LAGS_DAYS}
    valid = {lag: rho for lag, rho in observed.items() if np.isfinite(rho)}
    best_lag = max(valid, key=lambda lag: abs(valid[lag]))
    best_rho = valid[best_lag]

    n = len(y_rank)
    allowed_shifts = np.arange(min_shift, n - min_shift)
    null_peaks = np.empty(N_SURROGATES)
    for b in range(N_SURROGATES):
        y_perm = np.roll(y_rank, int(rng.choice(allowed_shifts)))
        null_peaks[b] = np.nanmax([abs(_lag_corr(x_rank, y_perm, lag)) for lag in LAGS_DAYS])
    p_value = (1.0 + np.sum(null_peaks >= abs(best_rho))) / (1.0 + N_SURROGATES)
    return best_lag, best_rho, p_value


def run_screening(df, min_shift):
    """Screen all hydroclimate x valuation pairs and adjust p-values with Benjamini-Hochberg FDR."""
    rank_cache = {f"{c}_anom": rank_array(df[f"{c}_anom"]) for c in PHYSICAL_COLUMNS}
    rank_cache.update({c: rank_array(df[c]) for c in VALUATION_RESPONSE_COLUMNS})
    rng = np.random.default_rng(SEED)

    rows = []
    for pc in PHYSICAL_COLUMNS:
        for rc in VALUATION_RESPONSE_COLUMNS:
            best_lag, rho, p = max_lag_spearman_test(rank_cache[f"{pc}_anom"], rank_cache[rc], min_shift, rng)
            rows.append({
                "physical_variable": pc, "physical_label": PHYSICAL_LABELS[pc],
                "valuation_response": rc, "valuation_label": VALUATION_LABELS[rc],
                "best_lag_days": best_lag, "spearman_rho": rho, "p_value_surrogate": p,
            })
    results = pd.DataFrame(rows)
    results["q_value_fdr"] = fdr_bh(results["p_value_surrogate"].to_numpy(dtype=float))
    return results.sort_values(["q_value_fdr", "p_value_surrogate"], ascending=[True, True]).reset_index(drop=True)


def lag_response_curve(df, physical_col, response_col="Forward_AbsAR_20d"):
    x_rank = rank_array(df[f"{physical_col}_anom"])
    y_rank = rank_array(df[response_col])
    return pd.DataFrame([{"physical_variable": physical_col, "lag_days": lag,
                          "spearman_rho": _lag_corr(x_rank, y_rank, lag)} for lag in LAGS_DAYS])


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------
def _savefig(fig, path):
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _set_year_axis(ax):
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


FIG_SERIES = [
    ("Adjusted_Close", "Adjusted close"), ("Relative_Valuation_Index", "Market-adjusted valuation index"),
    ("Abs_Abnormal_Return", "Absolute abnormal return"), ("Precipitation_mm", "Precipitation"),
    ("Runoff_Total_mm", "Total runoff"), ("Baseflow_mm", "Baseflow"),
    ("Root_Moisture_kg_m2", "Root-zone soil moisture"), ("Evapotranspiration_mm", "Evapotranspiration"),
    ("Temperature_C", "Temperature"), ("Water_Balance_mm", "Water balance"),
    ("Water_Availability_Index", "Water availability"), ("Dry_Heat_Stress", "Dry-heat stress"),
]


def plot_daily_series(df, fig_dir):
    fig, axes = plt.subplots(4, 3, figsize=(15.8, 11.5), sharex=True)
    for ax, (col, label) in zip(axes.ravel(), FIG_SERIES):
        s = df[col]
        color = COLORS.get(col, "#333333")
        ax.plot(s.index, s.values, color=color, lw=0.55, alpha=0.35)
        roll = s.rolling(20, min_periods=5).mean()
        ax.plot(roll.index, roll.values, color=color, lw=1.8)
        ax.set_title(label)
        ax.set_ylabel(UNITS.get(col, ""))
        _set_year_axis(ax)
        ax.grid(True)
    fig.suptitle("Daily time-synchronized valuation and basin hydroclimate records", fontsize=15, y=0.995)
    fig.autofmt_xdate()
    _savefig(fig, fig_dir / "fig3_daily_series.png")


def plot_monthly_climatology(df, fig_dir):
    months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    fig, axes = plt.subplots(4, 3, figsize=(16.2, 13.2))
    for ax, (col, label) in zip(axes.ravel(), FIG_SERIES):
        data = [df.loc[df.index.month == m, col].dropna().values for m in range(1, 13)]
        color = COLORS.get(col, "#5b8db8")
        bp = ax.boxplot(data, patch_artist=True, widths=0.62,
                        flierprops={"marker": ".", "markersize": 2.2, "markerfacecolor": color,
                                    "markeredgecolor": color, "alpha": 0.35},
                        medianprops={"color": "#222222", "linewidth": 1.1})
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.24)
            patch.set_edgecolor("#444444")
        ax.set_title(label)
        ax.set_ylabel(UNITS.get(col, "value"))
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(months, fontsize=8)
        ax.grid(True, axis="y")
    fig.suptitle("Monthly climatology of raw daily variables", fontsize=15, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    _savefig(fig, fig_dir / "fig4_monthly_climatology.png")


def plot_anomalies(df, fig_dir):
    fig, axes = plt.subplots(5, 2, figsize=(14.5, 13.0), sharex=True)
    for ax, col in zip(axes.ravel(), PHYSICAL_COLUMNS):
        s = df[f"{col}_anom"].rolling(10, min_periods=4).mean().clip(-4, 4)
        color = COLORS.get(col, "#333333")
        ax.axhline(0, color="#333333", lw=0.7)
        ax.fill_between(s.index, 0, s.values, where=s.values >= 0, color=color, alpha=0.18)
        ax.fill_between(s.index, 0, s.values, where=s.values < 0, color=color, alpha=0.08)
        ax.plot(s.index, s.values, color=color, lw=1.25)
        ax.set_title(PHYSICAL_LABELS[col])
        ax.set_ylabel("robust z")
        _set_year_axis(ax)
        ax.grid(True)
    fig.suptitle("Seasonally adjusted hydroclimate anomalies", fontsize=15, y=0.995)
    fig.autofmt_xdate()
    _savefig(fig, fig_dir / "fig5_anomalies.png")


def _spearman(a, b):
    joined = pd.concat([a, b], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(joined) < 8:
        return np.nan
    return pearson_corr(joined.iloc[:, 0].rank().to_numpy(), joined.iloc[:, 1].rank().to_numpy())


def plot_internal_correlation(df, fig_dir):
    labels = [PHYSICAL_LABELS[c] for c in PHYSICAL_COLUMNS]
    mat = np.full((len(PHYSICAL_COLUMNS), len(PHYSICAL_COLUMNS)), np.nan)
    for i, c1 in enumerate(PHYSICAL_COLUMNS):
        for j, c2 in enumerate(PHYSICAL_COLUMNS):
            mat[i, j] = _spearman(df[f"{c1}_anom"], df[f"{c2}_anom"])
    fig, ax = plt.subplots(figsize=(10.8, 8.8))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = mat[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8.5,
                    color="white" if abs(val) > 0.62 else "#222222")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Spearman rank correlation")
    ax.set_title("Internal correlation structure of basin hydroclimate anomalies", fontsize=14)
    _savefig(fig, fig_dir / "fig6_internal_correlation.png")


def plot_screening_matrix(results, fig_dir):
    """Screening matrix: peak Spearman rho and lead time per pair, with FDR significance marks."""
    rho = results.pivot(index="physical_variable", columns="valuation_response", values="spearman_rho")
    lag = results.pivot(index="physical_variable", columns="valuation_response", values="best_lag_days")
    q = results.pivot(index="physical_variable", columns="valuation_response", values="q_value_fdr")
    rho = rho.reindex(index=PHYSICAL_COLUMNS, columns=VALUATION_RESPONSE_COLUMNS)
    lag = lag.reindex(index=PHYSICAL_COLUMNS, columns=VALUATION_RESPONSE_COLUMNS)
    q = q.reindex(index=PHYSICAL_COLUMNS, columns=VALUATION_RESPONSE_COLUMNS)

    fig, ax = plt.subplots(figsize=(11.0, 9.0))
    im = ax.imshow(rho.to_numpy(dtype=float), cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    ax.set_xticks(np.arange(len(VALUATION_RESPONSE_COLUMNS)))
    ax.set_xticklabels([VALUATION_LABELS[c] for c in VALUATION_RESPONSE_COLUMNS], rotation=30, ha="right")
    ax.set_yticks(np.arange(len(PHYSICAL_COLUMNS)))
    ax.set_yticklabels([PHYSICAL_LABELS[c] for c in PHYSICAL_COLUMNS])
    for i, pc in enumerate(PHYSICAL_COLUMNS):
        for j, rc in enumerate(VALUATION_RESPONSE_COLUMNS):
            r = rho.loc[pc, rc]
            if not np.isfinite(r):
                continue
            qv = q.loc[pc, rc]
            mark = "**" if qv < 0.05 else ("*" if qv < 0.10 else "")
            ax.text(j, i, f"{r:.2f}{mark}\n{int(lag.loc[pc, rc])}d", ha="center", va="center",
                    fontsize=8.2, color="white" if abs(r) > 0.32 else "#222222")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Peak Spearman rho")
    ax.set_title("Lagged hydroclimate-valuation screening matrix\n(* q < 0.10; ** q < 0.05)", fontsize=13)
    fig.tight_layout()
    _savefig(fig, fig_dir / "fig7_screening_matrix.png")


def plot_lag_response(df, fig_dir):
    specs = [("Root_Moisture_kg_m2", "Root-zone soil moisture"), ("Baseflow_mm", "Baseflow")]
    fig, ax = plt.subplots(figsize=(11.8, 6.6))
    for col, label in specs:
        curve = lag_response_curve(df, col).sort_values("lag_days")
        color = COLORS[col]
        ax.plot(curve["lag_days"], curve["spearman_rho"], marker="o", ms=5.5, lw=2.2, color=color,
                label=f"{label} -> forward 20-day valuation stress")
        peak = curve.loc[curve["spearman_rho"].idxmax()]
        ax.scatter([peak["lag_days"]], [peak["spearman_rho"]], s=95, color="#d62728",
                   edgecolor="white", linewidth=1.2, zorder=6)
        ax.annotate(f"max rho={peak['spearman_rho']:.2f}\nlag={int(peak['lag_days'])}d",
                    xy=(peak["lag_days"], peak["spearman_rho"]),
                    xytext=(8, 12 if col == "Root_Moisture_kg_m2" else -34), textcoords="offset points",
                    fontsize=9, fontweight="bold", color="#7a1111",
                    bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#d62728", "alpha": 0.92})
    ax.axhline(0, color="#333333", lw=0.9)
    ax.set_xticks(LAGS_DAYS)
    ax.set_xlabel("Physical-variable lead time (trading days)")
    ax.set_ylabel("Spearman rho with forward 20-day valuation stress")
    ax.set_title("Lag-response curves for slow basin-storage variables", fontsize=13)
    ax.grid(True)
    ax.legend(loc="upper left")
    fig.tight_layout()
    _savefig(fig, fig_dir / "fig8_lag_response.png")


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--matrix", help="Path to a prepared analysis matrix CSV (skips extraction).")
    parser.add_argument("--finance", help="Path to a CSMAR price CSV (needed for a fresh build).")
    parser.add_argument("--out", default="./outputs", help="Output directory.")
    parser.add_argument("--gee-project", default=os.environ.get("GEE_PROJECT_ID", ""),
                        help="Google Earth Engine project id (or set GEE_PROJECT_ID).")
    parser.add_argument("--no-figures", action="store_true", help="Skip figure generation.")
    args = parser.parse_args()

    np.random.seed(SEED)
    out_dir = Path(args.out)
    fig_dir = out_dir / "figures"
    table_dir = out_dir / "tables"
    for d in (out_dir, fig_dir, table_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 1. Assemble the daily analysis matrix.
    if args.matrix:
        df = pd.read_csv(args.matrix, parse_dates=["Date"]).sort_values("Date").set_index("Date")
        df = df.replace([np.inf, -np.inf], np.nan)
        if "Water_Availability_Index" not in df.columns or "Dry_Heat_Stress" not in df.columns:
            df = add_physical_indexes(df)
    else:
        if not args.finance:
            parser.error("Provide --matrix, or --finance (with an Earth Engine project) to build from scratch.")
        climate = extract_climate_from_gee(args.gee_project, fig_dir)
        finance = load_financial_csv(args.finance)
        df = assemble_matrix(climate, finance)

    df = df.dropna(subset=["Log_Return", "Market_Log_Return"]).drop_duplicates()

    # 2. Build responses and anomalies (unless a fully prepared matrix already contains them).
    if "Forward_AbsAR_20d" not in df.columns or f"{PHYSICAL_COLUMNS[0]}_anom" not in df.columns:
        df, alpha, beta = build_features(df)
        print(f"Market model: alpha={alpha:.6f}, beta={beta:.3f}")
    df.to_csv(out_dir / "analysis_matrix.csv")
    print(f"Sample: {df.index.min().date()} to {df.index.max().date()}, n={len(df)} trading days")

    # 3. Lagged screening with circular-shift surrogates and FDR control.
    min_shift = surrogate_min_shift(df)
    print(f"Circular-shift minimum offset: {min_shift} trading days; surrogates: {N_SURROGATES}")
    results = run_screening(df, min_shift)
    results.to_csv(table_dir / "screening_results.csv", index=False)
    retained = results[results["q_value_fdr"] < 0.05]
    print(f"Pairs retained at q<0.05: {len(retained)}")
    print(retained[["physical_label", "valuation_label", "best_lag_days", "spearman_rho", "q_value_fdr"]]
          .to_string(index=False))

    for col in ("Root_Moisture_kg_m2", "Baseflow_mm"):
        lag_response_curve(df, col).to_csv(table_dir / f"lag_response_{col}.csv", index=False)

    # 4. Figures.
    if not args.no_figures:
        plot_daily_series(df, fig_dir)
        plot_monthly_climatology(df, fig_dir)
        plot_anomalies(df, fig_dir)
        plot_internal_correlation(df, fig_dir)
        plot_screening_matrix(results, fig_dir)
        plot_lag_response(df, fig_dir)
        print(f"Figures written to {fig_dir}")

    summary = {
        "company": COMPANY_LABEL, "ticker": TICKER, "market_ticker": MARKET_TICKER,
        "sample_start": str(df.index.min().date()), "sample_end": str(df.index.max().date()),
        "n_trading_days": int(len(df)), "lags_days": LAGS_DAYS, "n_surrogates": N_SURROGATES,
        "fdr_alpha": FDR_ALPHA, "n_pairs_q_lt_0.05": int(len(retained)),
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
