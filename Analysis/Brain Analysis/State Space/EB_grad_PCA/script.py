# =====================================================================
# Event-Structure Moderation of Thought→Brain Coupling (Full Pipeline)
# Author: Sam Ketcheson
# =====================================================================
# Inputs:
#   1) COMBINED_FILE: GradientSpace_vs_KDE_combined.csv
#        columns: Video, run_time, Speed_z, KDE_value   (per timepoint)
#   2) PCA_DIR: per-video *_PCA_timecourse.csv files
#        columns include: run_time and PCA components (e.g., Mean_PCA_1 ...)
#
# Analyses:
#   (1) Moderation LMM:       Speed_z ~ PCAk * KDE_z + (1|Video)
#   (2) Boundary-locked r(t): Rolling corr Speed_z↔PCAk around boundaries + circular-shift null
#   (3) Per-boundary Δ model: ΔSpeed ~ ΔPCAk + (1|Video)
#   (4) Variance partitioning across nested models (unique/shared)
#   (5) Event phase LMM:      Speed_z ~ PCAk * Phase + (1|Video)
#   (6) Mediation (explor.):  KDE_z → PCAk → Speed_z via block bootstrap
#   (7) Gradient-dim option:  |ΔGi| outcomes (requires gradient file; toggle)
#   (8) Predictive ridge:     Features [PCA, KDE, PCA×KDE], blocked CV by video
# =====================================================================

import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy.signal import find_peaks
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score

# ------------------ CONFIG ------------------
PCA_DIR = Path(r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\mDES_timecourses_byRunTime")
COMBINED_FILE = Path(r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\Brain Analysis\State Space\LLM\GradientSpace_vs_KDE_combined.csv")
OUTPUT_DIR = Path(r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\Brain Analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# (Optional) Gradient time-series for analysis 7:
HAVE_GRADIENTS = False
GRADIENT_FILE = Path(r"C:\Users\Smallwood Lab\Downloads\skipper_data_moviemanifold_5factor_gradient_Common_NativePCAs.csv")

# General params
TIME_TOL = 0.5       # seconds for merge_asof
FS = 1.0             # assumed timebase after your earlier processing (1 Hz)
ROLL_WIN = 11        # rolling correlation window (odd number, seconds)
BOUNDARY_PCT = 95    # percentile threshold to pick KDE peaks
MIN_PEAK_DIST = 3    # min seconds between peaks
DELTA_WIN = 5        # per-boundary Δ window (+/- seconds)
PHASE_EARLY = 5      # early after boundary (0..+5s)
PHASE_LATE = 5       # late pre boundary (-5..0s)
N_BOOT = 1000        # mediation block bootstrap samples
BLOCK_LEN = 10       # seconds per block in bootstrap
ALPHAS = (0.1, 1.0, 10.0, 100.0)  # ridge grid
np.random.seed(42)

# ------------------ HELPERS ------------------
def standardize_cols(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df

def zscore_within_group(df, group, cols):
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = df.groupby(group)[c].transform(lambda x: (x - x.mean()) / (x.std(ddof=0) if x.std(ddof=0) not in (0, np.nan) else 1.0))
    return df

def detect_pca_cols(cols):
    """Return dict like {'PCA1':'Mean_PCA_1', 'PCA2': 'pc2', ...} from any naming."""
    pmap = {}
    for c in cols:
        lc = c.lower().replace('-', '_').replace(' ', '_')
        m = re.search(r'(?:^|_)(?:mean_)?(?:native)?p?ca[_ ]?(\d+)$', lc)
        if m:
            k = int(m.group(1)); key = f'PCA{k}'
            pmap.setdefault(key, c)
        m2 = re.search(r'(?:^|_)pc(\d+)$', lc)
        if m2:
            k = int(m2.group(1)); key = f'PCA{k}'
            pmap.setdefault(key, c)
    return dict(sorted(pmap.items(), key=lambda kv: int(kv[0].replace('PCA',''))))

def load_merge_all():
    # Combined speed+KDE
    gk = pd.read_csv(COMBINED_FILE)
    gk = standardize_cols(gk)
    # Normalize names
    col_map = {}
    for c in gk.columns:
        lc = c.lower()
        if lc in ('video','film','movie'):
            col_map[c] = 'Video'
        elif lc in ('run_time','runtime','time','t'):
            col_map[c] = 'run_time'
        elif 'speed' in lc:
            col_map[c] = 'Speed_z' if 'z' in lc else 'Speed'
        elif 'kde' in lc or 'boundary' in lc:
            col_map[c] = 'KDE_value'
    gk = gk.rename(columns=col_map)
    if 'Speed_z' not in gk.columns and 'Speed' in gk.columns:
        gk = gk.rename(columns={'Speed':'Speed_z'})
    gk['Video'] = gk['Video'].astype(str).str.replace('.mp4','', regex=False).str.replace(' ','_', regex=False)
    gk['run_time'] = pd.to_numeric(gk['run_time'], errors='coerce').astype(float)

    # Merge with each PCA file
    merged = []
    for f in PCA_DIR.glob("*_PCA_timecourse.csv"):
        vname = f.stem.replace('_PCA_timecourse','').replace('.mp4','')
        vid_key = vname.replace(' ','_')
        pca = pd.read_csv(f)
        pca = standardize_cols(pca)
        # time col
        tcol = None
        for cand in ['run_time','runtime','time','t','seconds','sec', 'Run_time']:
            if cand in pca.columns:
                tcol = cand; break
        if tcol is None:
            print(f"⚠️ No time column in {f}, skipping.")
            continue
        pca = pca.rename(columns={tcol:'run_time'})
        pca['run_time'] = pd.to_numeric(pca['run_time'], errors='coerce').astype(float)
        pmap = detect_pca_cols(pca.columns)
        if not pmap:
            print(f"⚠️ No PCA columns in {f}, skipping.")
            continue
        pca = pca[['run_time'] + list(pmap.values())].dropna(subset=['run_time']).sort_values('run_time')
        sub = gk[gk['Video'].str.lower() == vid_key.lower()].sort_values('run_time')
        if sub.empty:
            print(f"⚠️ No combined rows for {vid_key}")
            continue
        m = pd.merge_asof(sub, pca, on='run_time', direction='nearest', tolerance=TIME_TOL)
        # rename PCA cols to canonical
        inv = {orig:canon for canon,orig in pmap.items()}
        m = m.rename(columns=inv)
        m['Video'] = vid_key
        # drop rows missing any PCA
        m = m.dropna(subset=list(pmap.keys()))
        merged.append(m)

    if not merged:
        raise RuntimeError("No merged data assembled. Check inputs.")
    df = pd.concat(merged, ignore_index=True)

    # z-score within video
    pca_cols = sorted([c for c in df.columns if re.fullmatch(r'PCA\d+', c)])
    df = zscore_within_group(df, 'Video', ['Speed_z','KDE_value'] + pca_cols)
    return df, pca_cols

def kde_peaks_for_video(kde_series, percentile=95, min_dist=3):
    x = kde_series.to_numpy()
    thr = np.nanpercentile(x, percentile)
    idx, _ = find_peaks(x, height=thr, distance=min_dist)
    return idx

def rolling_corr(a, b, win):
    # centered rolling correlation with odd window size
    n = len(a); w = win//2
    out = np.full(n, np.nan)
    for i in range(n):
        lo, hi = max(0, i-w), min(n, i+w+1)
        aa, bb = a[lo:hi], b[lo:hi]
        if len(aa) >= 5 and np.std(aa)>0 and np.std(bb)>0:
            out[i] = np.corrcoef(aa, bb)[0,1]
    return out

def circ_shift(x, k):
    k = int(k) % len(x)
    if k == 0: return x.copy()
    return np.concatenate([x[-k:], x[:-k]])

def signed_distance_to_boundaries(length, peaks):
    # distance (in seconds) to nearest past/next boundary
    dist = np.full(length, np.inf)
    for i in range(length):
        d = np.min(np.abs(peaks - i)) if len(peaks) else np.inf
        dist[i] = d
    return dist

def nearest_boundary_phase(idx, peaks, early=5, late=5):
    # label each index with phase wrt nearest boundary
    labels = []
    for i in range(idx):
        labels.append("Mid")
    labels = np.array(labels)  # not used (we compute with distances below)
    return labels

def block_bootstrap_indices(n, block_len, rng):
    # return a permutation of indices formed by block-wise circular sampling
    starts = rng.integers(0, n, size=int(np.ceil(n/block_len)))
    idx = []
    for s in starts:
        seg = np.arange(s, s+block_len) % n
        idx.append(seg)
    idx = np.concatenate(idx)[:n]
    return idx

# ------------------ LOAD & MERGE ------------------
print("📂 Loading & merging inputs …")
DF, PCA_COLS = load_merge_all()
print(f"✅ Merged rows: {len(DF)} across {DF['Video'].nunique()} videos")
print(f"🧠 PCA components detected: {PCA_COLS}")

# Make standardized column aliases
DF = DF.rename(columns={'run_time':'Time'})
DF['KDE_z'] = DF['KDE_value']  # already z-scored within video above

# Create output subfolders
(OUTPUT_DIR / "1_moderation").mkdir(exist_ok=True, parents=True)
(OUTPUT_DIR / "2_boundary_locked").mkdir(exist_ok=True, parents=True)
(OUTPUT_DIR / "3_delta").mkdir(exist_ok=True, parents=True)
(OUTPUT_DIR / "4_variance_partition").mkdir(exist_ok=True, parents=True)
(OUTPUT_DIR / "5_phase").mkdir(exist_ok=True, parents=True)
(OUTPUT_DIR / "6_mediation").mkdir(exist_ok=True, parents=True)
(OUTPUT_DIR / "8_predictive").mkdir(exist_ok=True, parents=True)

# ------------------ (1) MODERATION LMM ------------------
mod_rows = []
for p in PCA_COLS:
    print(f"\n[1] Moderation LMM for {p} …")
    use = DF[['Video','Time','Speed_z','KDE_z',p]].dropna()
    formula = f"Speed_z ~ {p} * KDE_z"
    try:
        m = smf.mixedlm(formula, use, groups=use['Video']).fit(reml=False)
        coefs = pd.DataFrame({
            'term': m.params.index,
            'estimate': m.params.values,
            'se': m.bse.values,
            't': m.tvalues.values,
            'p': m.pvalues.values
        })
        coefs.to_csv(OUTPUT_DIR / "1_moderation" / f"{p}_coefficients.csv", index=False)
        for _, r in coefs.iterrows():
            mod_rows.append({'PCA': p, **r.to_dict()})
    except Exception as e:
        with open(OUTPUT_DIR / "1_moderation" / f"{p}_ERROR.txt",'w') as fh:
            fh.write(str(e))
if mod_rows:
    pd.DataFrame(mod_rows).to_csv(OUTPUT_DIR / "1_moderation" / "summary_all_pca.csv", index=False)

# ------------------ (2) BOUNDARY-LOCKED ROLLING CORR ------------------
print("\n[2] Boundary-locked rolling correlation …")
win = ROLL_WIN if ROLL_WIN % 2 == 1 else ROLL_WIN + 1
half = win // 2
pad = 10  # seconds around boundary to plot/average
rng = np.random.default_rng(123)

for p in PCA_COLS:
    curves = []
    null_lo, null_hi = [], []
    for vid, sub in DF.groupby('Video'):
        sub = sub.sort_values('Time')
        # boundary peaks from KDE_value (pre z-scoring effect OK)
        peaks = kde_peaks_for_video(sub['KDE_value'], percentile=BOUNDARY_PCT, min_dist=MIN_PEAK_DIST)
        if len(peaks) < 3:  # need some events
            continue
        r = rolling_corr(sub['Speed_z'].to_numpy(), sub[p].to_numpy(), win=win)
        # extract windows around peaks
        for pk in peaks:
            lo, hi = pk - pad, pk + pad
            if lo < 0 or hi >= len(r): continue
            curves.append(r[lo:hi+1])

        # circular-shift null envelope
        # 100 random shifts:
        null_mat = []
        for _ in range(100):
            k = rng.integers(0, len(sub))
            rp = rolling_corr(sub['Speed_z'].to_numpy(), circ_shift(sub[p].to_numpy(), k), win=win)
            for pk in peaks:
                lo, hi = pk - pad, pk + pad
                if lo < 0 or hi >= len(rp): continue
                null_mat.append(rp[lo:hi+1])
        if null_mat:
            null_mat = np.vstack(null_mat)
            null_lo.append(np.nanpercentile(null_mat, 2.5, axis=0))
            null_hi.append(np.nanpercentile(null_mat, 97.5, axis=0))

    if not curves:
        continue
    curves = np.vstack(curves)
    mean_curve = np.nanmean(curves, axis=0)
    tvec = np.arange(-pad, pad+1)

    # null bands
    if null_lo and null_hi:
        lo_band = np.nanmean(np.vstack(null_lo), axis=0)
        hi_band = np.nanmean(np.vstack(null_hi), axis=0)
    else:
        lo_band = hi_band = None

    outp = OUTPUT_DIR / "2_boundary_locked" / f"{p}_boundary_locked_corr.csv"
    pd.DataFrame({'t': tvec, 'mean_r': mean_curve,
                  'null_lo': lo_band if lo_band is not None else np.full_like(tvec, np.nan),
                  'null_hi': hi_band if hi_band is not None else np.full_like(tvec, np.nan)}).to_csv(outp, index=False)

    plt.figure(figsize=(7,4))
    plt.plot(tvec, mean_curve, lw=2)
    if lo_band is not None:
        plt.fill_between(tvec, lo_band, hi_band, alpha=0.2, label='shifted null 95%')
    plt.axvline(0, ls='--', lw=1)
    plt.xlabel("Time from boundary (s)")
    plt.ylabel(f"Rolling corr r(Speed_z, {p})")
    plt.title(f"Boundary-locked coupling: {p}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "2_boundary_locked" / f"{p}_boundary_locked_corr.png", dpi=300)
    plt.close()

# ------------------ (3) PER-BOUNDARY Δ COUPLING ------------------
print("\n[3] Per-boundary Δ coupling …")
delta_rows = []
win = DELTA_WIN
for p in PCA_COLS:
    per_boundary = []
    for vid, sub in DF.groupby('Video'):
        sub = sub.sort_values('Time').reset_index(drop=True)
        peaks = kde_peaks_for_video(sub['KDE_value'], percentile=BOUNDARY_PCT, min_dist=MIN_PEAK_DIST)
        for pk in peaks:
            lo_pre, hi_pre = max(0, pk-win), pk
            lo_post, hi_post = pk+1, min(len(sub)-1, pk+win)
            if hi_post - lo_post < 2 or hi_pre - lo_pre < 2: 
                continue
            d_speed = sub.loc[lo_post:hi_post,'Speed_z'].mean() - sub.loc[lo_pre:hi_pre,'Speed_z'].mean()
            d_p = sub.loc[lo_post:hi_post,p].mean() - sub.loc[lo_pre:hi_pre,p].mean()
            per_boundary.append({'Video': vid, 'ΔSpeed': d_speed, f'Δ{p}': d_p})
    if not per_boundary:
        continue
    B = pd.DataFrame(per_boundary)
    # mixed model ΔSpeed ~ ΔP + (1|Video)
    m = smf.mixedlm(f"ΔSpeed ~ Δ{p}", B, groups=B['Video']).fit(reml=False)
    coef = pd.DataFrame({
        'term': m.params.index, 'estimate': m.params.values,
        'se': m.bse.values, 't': m.tvalues.values, 'p': m.pvalues.values
    })
    coef.to_csv(OUTPUT_DIR / "3_delta" / f"{p}_delta_coefficients.csv", index=False)
    B.to_csv(OUTPUT_DIR / "3_delta" / f"{p}_delta_points.csv", index=False)
    delta_rows.append({'PCA': p, **{k:coef.loc[coef['term']==k,'estimate'].values[0] for k in coef['term']}})

# ------------------ (4) VARIANCE PARTITIONING ------------------
print("\n[4] Variance partitioning …")
def pseudo_r2(formula, data, group):
    # Fit and compute R2 against intercept-only residual baseline on same rows
    rows = data.dropna()
    m_full = smf.mixedlm(formula, rows, groups=rows[group]).fit(reml=False)
    y = rows[formula.split('~')[0].strip()]
    yhat = m_full.fittedvalues
    rss = np.sum((y - yhat)**2)
    tss = np.sum((y - y.mean())**2)
    r2 = 1 - rss / (tss + 1e-12)
    return r2, m_full

# Build formula parts dynamically
p_terms = " + ".join(PCA_COLS)
DF_clean = DF[['Video','Speed_z','KDE_z'] + PCA_COLS].dropna()

R2_M0 = pseudo_r2("Speed_z ~ 1", DF_clean, "Video")[0]
R2_M1, m1 = pseudo_r2("Speed_z ~ KDE_z", DF_clean, "Video")
R2_M2, m2 = pseudo_r2(f"Speed_z ~ {p_terms}", DF_clean, "Video")
R2_M3, m3 = pseudo_r2(f"Speed_z ~ KDE_z + {p_terms}", DF_clean, "Video")

unique_KDE   = max(0.0, R2_M3 - R2_M2)
unique_PCA   = max(0.0, R2_M3 - R2_M1)
shared       = max(0.0, R2_M1 + R2_M2 - R2_M3)
vp = pd.DataFrame([{
    'R2_M0': R2_M0, 'R2_M1_KDE': R2_M1, 'R2_M2_PCA': R2_M2, 'R2_M3_both': R2_M3,
    'Unique_KDE': unique_KDE, 'Unique_PCA': unique_PCA, 'Shared': shared
}])
vp.to_csv(OUTPUT_DIR / "4_variance_partition" / "variance_partition.csv", index=False)

plt.figure(figsize=(5,4))
parts = ['Unique_KDE','Shared','Unique_PCA']
vals = [unique_KDE, shared, unique_PCA]
plt.bar(parts, vals)
plt.ylabel("Pseudo R²")
plt.title("Variance partitioning: Speed_z")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "4_variance_partition" / "variance_partition.png", dpi=300)
plt.close()

# ------------------ (5) EVENT PHASE ANALYSIS ------------------
print("\n[5] Event phase LMM …")
def label_phase(kde, early=PHASE_EARLY, late=PHASE_LATE):
    x = kde.to_numpy()
    peaks, _ = find_peaks(x, height=np.nanpercentile(x, BOUNDARY_PCT), distance=MIN_PEAK_DIST)
    phase = np.array(['Mid']*len(x), dtype=object)
    for pk in peaks:
        # late pre-boundary window
        lo = max(0, pk-late); hi = pk
        phase[lo:hi] = 'LatePre'
        # early after boundary window
        lo = pk; hi = min(len(x), pk+early)
        phase[lo:hi] = 'EarlyPost'
    return phase

# Replace the old line with this block
phase_labels = []
for vid, sub in DF.groupby("Video"):
    labels = label_phase(sub["KDE_value"])
    phase_labels.append(pd.Series(labels, index=sub.index))
DF["Phase"] = pd.concat(phase_labels).sort_index()

phase_rows = []
for p in PCA_COLS:
    use = DF[['Video','Speed_z','KDE_z','Phase',p]].dropna()
    use['Phase'] = use['Phase'].astype('category')
    try:
        m = smf.mixedlm(f"Speed_z ~ {p} * Phase", use, groups=use['Video']).fit(reml=False)
        coefs = pd.DataFrame({
            'term': m.params.index, 'estimate': m.params.values,
            'se': m.bse.values, 't': m.tvalues.values, 'p': m.pvalues.values
        })
        coefs.to_csv(OUTPUT_DIR / "5_phase" / f"{p}_phase_coefficients.csv", index=False)
    except Exception as e:
        with open(OUTPUT_DIR / "5_phase" / f"{p}_ERROR.txt",'w') as fh:
            fh.write(str(e))

# ------------------ (6) MEDIATION (Exploratory) ------------------
print("\n[6] Mediation (exploratory bootstrap) …")
# For each PCAk: a: PCAk ~ KDE_z; b: Speed_z ~ KDE_z + PCAk; indirect = a*b via block bootstrap
med_rows = []
rng = np.random.default_rng(7)
for p in PCA_COLS:
    use = DF[['Video','Speed_z','KDE_z',p]].dropna().copy()
    # fit on original
    a_mod = smf.mixedlm(f"{p} ~ KDE_z", use, groups=use['Video']).fit(reml=False)
    b_mod = smf.mixedlm(f"Speed_z ~ KDE_z + {p}", use, groups=use['Video']).fit(reml=False)
    a = a_mod.params.get('KDE_z', np.nan)
    b = b_mod.params.get(p, np.nan)
    indirect = a * b

    # block bootstrap by video
    boots = []
    for _ in range(N_BOOT):
        tmp_list = []
        for vid, sub in use.groupby('Video'):
            n = len(sub)
            idx = block_bootstrap_indices(n, BLOCK_LEN, rng)
            tmp_list.append(sub.iloc[idx].assign(Video=vid))
        bs = pd.concat(tmp_list, ignore_index=True)
        try:
            a_b = smf.mixedlm(f"{p} ~ KDE_z", bs, groups=bs['Video']).fit(reml=False).params.get('KDE_z', np.nan)
            b_b = smf.mixedlm(f"Speed_z ~ KDE_z + {p}", bs, groups=bs['Video']).fit(reml=False).params.get(p, np.nan)
            boots.append(a_b * b_b)
        except Exception:
            continue
    boots = np.array(boots)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5]) if len(boots) else (np.nan, np.nan)
    med_rows.append({'PCA': p, 'a': a, 'b': b, 'indirect': indirect, 'boot_lo': lo, 'boot_hi': hi})

pd.DataFrame(med_rows).to_csv(OUTPUT_DIR / "6_mediation" / "mediation_summary.csv", index=False)

# ------------------ (7) GRADIENT DIMENSION SPECIFICITY (optional) ------------------
if HAVE_GRADIENTS:
    print("\n[7] Gradient-dimension specificity …")
    # Load gradient coords over time, resample to 1 Hz, merge and compute |ΔGi|
    # (This block requires your earlier resampling code; omitted for brevity.)
    pass

# ------------------ (8) PREDICTIVE CHECK (ridge, blocked CV by movie) ------------------
print("\n[8] Predictive check (ridge) …")
X_list, y_list, g_list = [], [], []
# Features: all PCA + KDE + PCA×KDE
feat_cols = PCA_COLS + ['KDE_z'] + [f"{p}:KDE" for p in PCA_COLS]
tmp = DF[['Video','Speed_z','KDE_z'] + PCA_COLS].dropna().copy()
for p in PCA_COLS:
    tmp[f"{p}:KDE"] = tmp[p] * tmp['KDE_z']

X = tmp[feat_cols].to_numpy()
y = tmp['Speed_z'].to_numpy()
groups = tmp['Video'].to_numpy()

gkf = GroupKFold(n_splits=max(2, min(5, len(np.unique(groups)))))
ridge = RidgeCV(alphas=ALPHAS, store_cv_values=False)
preds, trues = [], []
for train_idx, test_idx in gkf.split(X, y, groups):
    ridge.fit(X[train_idx], y[train_idx])
    yhat = ridge.predict(X[test_idx])
    preds.append(yhat); trues.append(y[test_idx])
yhat = np.concatenate(preds); ytrue = np.concatenate(trues)
r2_full = r2_score(ytrue, yhat)

# Compare without interactions
feat_cols_noint = PCA_COLS + ['KDE_z']
X2 = tmp[feat_cols_noint].to_numpy()
preds2, trues2 = [], []
for train_idx, test_idx in gkf.split(X2, y, groups):
    ridge.fit(X2[train_idx], y[train_idx])
    yhat = ridge.predict(X2[test_idx])
    preds2.append(yhat); trues2.append(y[test_idx])
r2_noint = r2_score(np.concatenate(trues2), np.concatenate(preds2))

pd.DataFrame([{'model':'with_interactions','R2':r2_full},
              {'model':'no_interactions','R2':r2_noint}]).to_csv(
    OUTPUT_DIR / "8_predictive" / "ridge_blocked_cv_r2.csv", index=False
)

print("\n✅ Pipeline complete.")
print(f"All outputs in:\n{OUTPUT_DIR}")
