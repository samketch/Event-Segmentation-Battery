import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.stats import zscore
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# LOAD
# ----------------------------------------------------------
df = pd.read_csv(
    r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\Clean_Analysis\Within_Between\CSV_Within_Across\PCA_withinAcross_timepoints.csv"
)
bounds = pd.read_csv(
    r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\event_seg analysis\Analyzed_Data\Kernal\KDE_Results\all_videos_consensus_boundaries.csv"
)

grad_cols = [f"Mean_gradient_{i}" for i in range(1, 6)]

# ----------------------------------------------------------
# MARK BOUNDARY WINDOW
# Use a window that matches your sampling step (looks like 10 s)
# ----------------------------------------------------------
WIN = 5  # try 5 or 10 if your timepoints are 10 s apart
df["BoundaryHit"] = 0

for vid, sub in bounds.groupby("VideoName"):
    times = sub["ConsensusTime(s)"].to_numpy()
    mask_vid = df["VideoName"].eq(vid)
    t = df.loc[mask_vid, "Run_time"].to_numpy()
    hit = np.zeros(mask_vid.sum(), dtype=bool)
    for bt in times:
        hit |= np.abs(t - bt) <= WIN
    df.loc[mask_vid, "BoundaryHit"] = hit.astype(int)

# ----------------------------------------------------------
# Z-SCORE GRADIENTS WITHIN VIDEO
# ----------------------------------------------------------
for g in grad_cols:
    df[g + "_z"] = df.groupby("VideoName")[g].transform(lambda x: zscore(x, nan_policy="omit"))

df = df.dropna(subset=["BoundaryDensity"] + [g + "_z" for g in grad_cols]).reset_index(drop=True)
df["VideoName"] = df["VideoName"].astype("category")

# ----------------------------------------------------------
# SUMMARY: MEAN POSITION AT BOUNDARY VS NON-BOUNDARY
# ----------------------------------------------------------
summary = df.groupby("BoundaryHit")[grad_cols].mean()
print("\nMean gradient position by BoundaryHit:")
print(summary)

# ----------------------------------------------------------
# LOGISTIC REGRESSION WITH CLUSTERED SE BY VIDEO
# (Binomial GLM with cluster-robust SEs)
# ----------------------------------------------------------
formula_bin = "BoundaryHit ~ " + " + ".join([f"{g}_z" for g in grad_cols])
glm = smf.glm(formula_bin, data=df, family=sm.families.Binomial())
fit_glm = glm.fit(cov_type="cluster", cov_kwds={"groups": df["VideoName"]})
print("\nLogistic GLM with clustered SE by Video:")
print(fit_glm.summary())

# ----------------------------------------------------------
# LMM: CONTINUOUS BOUNDARY DENSITY ~ GRADIENTS + (1|VIDEO)
# ----------------------------------------------------------
df["BoundaryDensity_scaled"] = df["BoundaryDensity"] * 1000
formula_lmm = "BoundaryDensity_scaled ~ " + " + ".join([f"{g}_z" for g in grad_cols])

try:
    lmm = smf.mixedlm(formula_lmm, data=df, groups=df["VideoName"]).fit(method="lbfgs", reml=False)
    print("\nLMM: BoundaryDensity ~ gradients + (1|Video):")
    print(lmm.summary())
except Exception as e:
    print("\nLMM failed, falling back to OLS:")
    print(e)
    ols = smf.ols(formula_lmm, data=df).fit(cov_type="cluster", cov_kwds={"groups": df["VideoName"]})
    print(ols.summary())

# ----------------------------------------------------------
# OCCUPANCY HEATMAP IN G1-G2 SPACE
# ----------------------------------------------------------
def occupancy_heat(dd, gx, gy, n=40):
    x = dd[gx + "_z"].to_numpy()
    y = dd[gy + "_z"].to_numpy()
    z = dd["BoundaryDensity"].to_numpy()
    H = np.zeros((n, n), dtype=float)
    C = np.zeros((n, n), dtype=int)
    xe = np.linspace(np.nanpercentile(x, 1), np.nanpercentile(x, 99), n + 1)
    ye = np.linspace(np.nanpercentile(y, 1), np.nanpercentile(y, 99), n + 1)
    xi = np.digitize(x, xe) - 1
    yi = np.digitize(y, ye) - 1
    ok = (xi >= 0) & (xi < n) & (yi >= 0) & (yi < n)
    for i, j, val in zip(xi[ok], yi[ok], z[ok]):
        H[j, i] += val
        C[j, i] += 1
    M = np.divide(H, C, out=np.full_like(H, np.nan, dtype=float), where=C > 0)
    return xe, ye, M

xe, ye, M = occupancy_heat(df, "Mean_gradient_1", "Mean_gradient_2", n=40)
plt.figure(figsize=(6, 5))
plt.imshow(M, origin="lower", aspect="auto", extent=[xe[0], xe[-1], ye[0], ye[-1]])
plt.xlabel("G1 (z)")
plt.ylabel("G2 (z)")
plt.title("Mean BoundaryDensity in G1–G2 space")
plt.colorbar(label="BoundaryDensity")
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# MATCHED CONTROLS PER VIDEO, SIMPLE T-TEST ON G1
# ----------------------------------------------------------
controls, cases = [], []
for vid, sub in df.groupby("VideoName"):
    case = sub.loc[sub["BoundaryHit"] == 1, grad_cols].copy()
    if len(case) == 0:
        continue
    ctrl = sub.loc[sub["BoundaryHit"] == 0, grad_cols].sample(n=len(case), replace=False, random_state=1)
    cases.append(case.assign(VideoName=vid, Label="Boundary"))
    controls.append(ctrl.assign(VideoName=vid, Label="Control"))

if cases and controls:
    dd = pd.concat([pd.concat(cases), pd.concat(controls)], ignore_index=True)
    from scipy.stats import ttest_ind
    t, p = ttest_ind(
        dd.loc[dd.Label == "Boundary", "Mean_gradient_1"],
        dd.loc[dd.Label == "Control", "Mean_gradient_1"],
        equal_var=False,
    )
    print(f"\nBoundary vs control on Mean_gradient_1: t={t:.3f}, p={p:.4f}")

# ----------------------------------------------------------
# STRATIFY BY BOUNDARY DENSITY QUANTILES
# ----------------------------------------------------------
df["BD_bin"] = pd.qcut(df["BoundaryDensity"], q=[0, 0.5, 0.75, 0.9, 1.0], labels=["low", "med", "high", "peak"], duplicates="drop")
group_means = df.groupby("BD_bin")[grad_cols].mean()
print("\nGradient means by BoundaryDensity bin:")
print(group_means)

# ----------------------------------------------------------
# EVENT-LEVEL ANALYSES
# ----------------------------------------------------------
events_path = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\Clean_Analysis\Gradient_analysis\Event_length\Gradient_EventLength_perEvent.csv"
if os.path.exists(events_path):
    events = pd.read_csv(events_path)
    # z-score gradients within video for events
    for g in grad_cols:
        events[g + "_z"] = events.groupby("VideoName")[g].transform(lambda x: zscore(x, nan_policy="omit"))
    # length bins
    events["LenBin"] = pd.qcut(events["EventLength"], 3, labels=["short", "mid", "long"], duplicates="drop")
    print("\nEvent centroid means by length bin:")
    print(events.groupby("LenBin")[grad_cols].mean())

    # multinomial logit on length bins
    mod = smf.mnlogit("LenBin ~ " + " + ".join([f"{g}_z" for g in grad_cols]), data=events).fit()
    print("\nMultinomial logit: LenBin ~ gradients")
    print(mod.summary())

# ----------------------------------------------------------
# PERMUTATION NULL VIA WITHIN-VIDEO CIRCULAR SHIFTS
# ----------------------------------------------------------
rng = np.random.default_rng(1)

def circ_shift(a, k):
    k = int(k) % len(a)
    return np.concatenate([a[-k:], a[:-k]]) if len(a) else a

def permute_beta(dd, target="Mean_gradient_4_z", nperm=1000):
    betas = []
    formula = "BoundaryDensity_perm ~ " + " + ".join([f"{g}_z" for g in grad_cols])
    for _ in range(nperm):
        tmp = dd.copy()
        for vid, sub in tmp.groupby("VideoName"):
            if len(sub) < 2:
                continue
            k = rng.integers(1, len(sub))
            tmp.loc[sub.index, "BoundaryDensity_perm"] = circ_shift(sub["BoundaryDensity"].to_numpy(), k)
        fit = smf.ols(formula, data=tmp.dropna(subset=["BoundaryDensity_perm"])).fit()
        betas.append(fit.params.get(target, np.nan))
    return np.array(betas)

# Example use:
# null_betas = permute_beta(df, target="Mean_gradient_4_z", nperm=2000)
# print(np.nanmean(null_betas), np.nanstd(null_betas))

import statsmodels.formula.api as smf

# ----------------------------------------------------------
# Test whether BoundaryDensity predicts each gradient position
# ----------------------------------------------------------
results = []
for i in range(1, 6):
    g = f"Mean_gradient_{i}_z"
    formula = f"{g} ~ BoundaryDensity"
    try:
        m = smf.mixedlm(formula, data=df, groups=df["VideoName"])
        fit = m.fit(method="lbfgs", reml=False)
        results.append([g, fit.params["Intercept"], fit.params["BoundaryDensity"], fit.pvalues["BoundaryDensity"]])
        print(f"{g}: β={fit.params['BoundaryDensity']:.4f}, p={fit.pvalues['BoundaryDensity']:.4f}")
    except Exception as e:
        print(f"{g} failed ({e}); running OLS instead.")
        import statsmodels.api as sm
        fit = sm.OLS.from_formula(formula, data=df).fit(cov_type="cluster", cov_kwds={"groups": df["VideoName"]})
        results.append([g, fit.params["Intercept"], fit.params["BoundaryDensity"], fit.pvalues["BoundaryDensity"]])
        print(f"{g}: β={fit.params['BoundaryDensity']:.4f}, p={fit.pvalues['BoundaryDensity']:.4f}")

# Save summary table
out_table = pd.DataFrame(results, columns=["Gradient", "Intercept", "Beta_BoundaryDensity", "p_value"])
out_path = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\Clean_Analysis\Gradient_analysis\Events_In_gradients\BoundaryDensity_predicts_Gradients.csv"
out_table.to_csv(out_path, index=False)
print(f"\n✅ Saved gradient-prediction summary to:\n{out_path}")
print(out_table)

