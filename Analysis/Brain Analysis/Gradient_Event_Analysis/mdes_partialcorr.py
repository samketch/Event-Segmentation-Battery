"""
mdes_partialcorr.py
Relate MDES components to gradient speed while controlling for event KDE (boundary likelihood).
Uses MixedLM (random intercepts by participant) when possible, else GLS with AR(1).

Expected columns:
 - MDES CSV: ['movie','time','participant'(optional), 'Mean_PCA_1', 'Mean_PCA_2', ...]
 - Grad CSV: ['movie'(optional), 'time'(optional), gradient columns...]
 - Event KDE: ['movie'(optional), 'time','BoundaryDensity']

Outputs:
 - mdes_partialcorr_results.csv      (per movie × component; betas, SEs, p-values)
"""

import os
import re
import numpy as np
import pandas as pd
from scipy.stats import zscore
from scipy.ndimage import gaussian_filter1d

import statsmodels.api as sm
from statsmodels.regression.linear_model import GLS
from statsmodels.genmod.cov_struct import Autoregressive

# ---------------------------------------------------------------------
def find_mdes_components(df):
    patt = re.compile(r'(PCA|Mean_PCA|Component|PC)[ _-]*\d+', re.I)
    return [c for c in df.columns if patt.search(c)]

def build_speed_series(grad_df, gcols, smooth_sigma, fs):
    G = grad_df[gcols].values
    dG = np.vstack([G[1:] - G[:-1], G[-1:] - G[-2:-1]])
    speed = np.sqrt((dG ** 2).sum(axis=1))
    speed = gaussian_filter1d(zscore(speed), sigma=smooth_sigma)
    grad_df = grad_df.copy()
    grad_df["grad_speed"] = speed
    return grad_df

def try_mixedlm(df, dep, indep_cols, group_col):
    """MixedLM with random intercepts; returns (params, bse, pvalues) or None if it fails."""
    try:
        X = sm.add_constant(df[indep_cols])
        y = df[dep]
        md = sm.MixedLM(y, X, groups=df[group_col])
        mdf = md.fit(method="lbfgs", reml=False, maxiter=200, disp=False)
        return mdf.params, mdf.bse, mdf.pvalues
    except Exception as e:
        print(f"⚠️ MixedLM failed ({dep}): {e}")
        return None

def gls_ar1(df, dep, indep_cols, movie_col="movie"):
    """
    Fit GLS with AR(1) errors separately per movie, then combine via inverse-variance weighting.
    """
    rows = []
    for mv, sub in df.groupby(movie_col):
        if len(sub) < 10:
            continue
        X = sm.add_constant(sub[indep_cols])
        y = sub[dep].values
        # Estimate AR(1) rho via Yule-Walker-ish approach on residuals of OLS prefit
        pre = sm.OLS(y, X).fit()
        resid = pre.resid
        rho = np.corrcoef(resid[:-1], resid[1:])[0,1]
        # Clip rho to plausible range
        rho = float(np.clip(rho, -0.95, 0.95))
        # GLSAR from statsmodels (older API)
        gls = sm.GLSAR(y, X, rho)
        res = gls.iterative_fit(3)
        rows.append({"movie": mv, "params": res.params, "bse": res.bse, "pvalues": res.pvalues})

    if not rows:
        return None

    # Combine across movies by inverse-variance weighting (fixed-effects meta-analytic style)
    keys = rows[0]["params"].index
    pooled = {}
    for k in keys:
        ests, vars_ = [], []
        for r in rows:
            est = r["params"][k]
            var = (r["bse"][k] ** 2)
            if np.isfinite(est) and np.isfinite(var) and var > 0:
                ests.append(est); vars_.append(var)
        if not ests:
            pooled[k] = (np.nan, np.nan, np.nan)
            continue
        w = 1 / np.array(vars_)
        est_pool = np.sum(w * np.array(ests)) / np.sum(w)
        se_pool = np.sqrt(1 / np.sum(w))
        # Wald z
        z = est_pool / se_pool if se_pool > 0 else np.nan
        p = 2 * (1 - sm.stats.norm.cdf(abs(z))) if np.isfinite(z) else np.nan
        pooled[k] = (est_pool, se_pool, p)

    # return as Series-like dicts
    params = pd.Series({k: pooled[k][0] for k in keys})
    bse    = pd.Series({k: pooled[k][1] for k in keys})
    pvals  = pd.Series({k: pooled[k][2] for k in keys})
    return params, bse, pvals

# ---------------------------------------------------------------------
def run_mdes_partialcorr(cfg):
    print("→ Running MDES × gradient-speed partial correlation (controlling for boundary KDE)")

    grad = pd.read_csv(cfg["gradient_csv"])
    events = pd.read_csv(cfg["event_kde_csv"])
    mdes = pd.read_csv(cfg["mdes_csv"])

    gcols = cfg["gradient_cols"]
    fs = cfg.get("sampling_rate", 1.0)
    smooth = cfg.get("smoothing_sigma", 3)

    # infer join keys
    join_keys = [k for k in ["movie","time"] if k in grad.columns and k in events.columns]
    if not join_keys:
        # fallback: align by index/time only
        grad["time"] = np.arange(len(grad))
        events["time"] = np.arange(len(events))
        join_keys = ["time"]

    # add speed to gradient df
    grad = build_speed_series(grad, gcols, smooth, fs)

    # bring in boundary KDE
    ev_cols = join_keys + [c for c in events.columns if c not in join_keys]
    df = pd.merge(grad[join_keys + ["grad_speed"]], events[ev_cols], on=join_keys, how="inner")

    # bring in MDES (keep components only)
    mdes_comp_cols = find_mdes_components(mdes)
    keep_cols = join_keys + (["participant"] if "participant" in mdes.columns else []) + mdes_comp_cols
    df = pd.merge(df, mdes[keep_cols], on=join_keys, how="inner")

    # z-score predictors
    if "BoundaryDensity" not in df.columns:
        raise ValueError("event_kde_csv must include 'BoundaryDensity' column")
    df["grad_speed_z"] = zscore(df["grad_speed"].values)
    df["kde_z"] = zscore(df["BoundaryDensity"].values)
    df["speed_x_kde"] = df["grad_speed_z"] * df["kde_z"]

    have_participant = "participant" in df.columns
    have_movie = "movie" in df.columns

    results = []
    for comp in mdes_comp_cols:
        # dependent variable (optionally difference score if you prefer change)
        y = zscore(df[comp].values)
        df_model = df.copy()
        df_model[comp + "_z"] = y

        dep = comp + "_z"
        indeps = ["grad_speed_z", "kde_z", "speed_x_kde"]

        if have_participant:
            # MixedLM with random intercepts by participant (and optional movie fixed effects)
            Xcols = indeps + (["movie"] if have_movie else [])
            if have_movie:
                # dummy-code movie as fixed effects (drop first to avoid collinearity)
                dummies = pd.get_dummies(df_model["movie"], prefix="mv", drop_first=True)
                X = pd.concat([df_model[indeps], dummies], axis=1)
                df_fit = pd.concat([df_model[[dep, "participant"]], X], axis=1).dropna()
                params = try_mixedlm(df_fit, dep, list(X.columns), "participant")
            else:
                df_fit = df_model[[dep] + indeps + ["participant"]].dropna()
                params = try_mixedlm(df_fit, dep, indeps, "participant")

            if params is None:
                # fall back to GLS AR(1) per movie
                if have_movie:
                    params = gls_ar1(df_model[[dep] + indeps + ["movie"]].dropna(), dep, indeps, movie_col="movie")
                else:
                    # one big series
                    params = gls_ar1(df_model[[dep] + indeps].dropna().assign(movie="all"), dep, indeps, movie_col="movie")

        else:
            # no participant column → GLS(AR1) per movie (or single movie)
            if have_movie:
                params = gls_ar1(df_model[[dep] + indeps + ["movie"]].dropna(), dep, indeps, movie_col="movie")
            else:
                params = gls_ar1(df_model[[dep] + indeps].dropna().assign(movie="all"), dep, indeps, movie_col="movie")

        if params is None:
            print(f"⚠️ Could not fit model for {comp}. Skipping.")
            continue

        est, se, p = params
        # collect only the core effects (plus intercept if present)
        for term in est.index:
            results.append({
                "component": comp,
                "term": term,
                "beta": float(est[term]),
                "se": float(se[term]),
                "p": float(p[term]) if np.isfinite(p[term]) else np.nan
            })

    res_df = pd.DataFrame(results)
    # simple FDR across all tests
    if not res_df.empty:
        res_df = res_df.sort_values("p")
        m = len(res_df)
        res_df["q"] = (res_df["p"] * m / (np.arange(m) + 1)).clip(upper=1.0)
        res_df = res_df.sort_values(["component","term"])

    outdir = os.path.join(cfg["output_root"], "MDES")
    os.makedirs(outdir, exist_ok=True)
    res_df.to_csv(os.path.join(outdir, "mdes_partialcorr_results.csv"), index=False)

    print("✓ Saved MDES × gradient-speed (partial) results.\n")
