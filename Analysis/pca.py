import os
import pandas as pd
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
from factor_analyzer.rotator import Rotator
from sklearn.decomposition import PCA
import numpy as np


data = pd.read_csv('Analysis/output.csv').drop(["Task_name","Participant #"],axis=1)

bart = calculate_bartlett_sphericity(data)
kmo = calculate_kmo(data)
n_comp = 4
PCAmodel = PCA(n_components=n_comp)
rot = Rotator()

PCAmodel.fit(data)

loadings = PCAmodel.components_
loadings = rot.fit_transform(PCAmodel.components_.T).T
#lods = rot.transform(PCAmodel.components_.T)
names = data.columns
#loadings = PCAmodel.components_
loadings = pd.DataFrame(
    np.round(loadings.T, 3),
    index=names,
    columns=[f"Component {x}" for x in range(n_comp)],
)
PCAresults = np.dot(data,loadings).T
scores = ['0','1','2','3']
numloadings=5
for score in scores:
    
    try:
        loading = loadings[f"Component {score}"]
        loadingpos = loading.apply(lambda x: np.abs(x)).sort_values(ascending=False)
        tops = loading[loadingpos[:numloadings].index]
    except Exception:
        tops = "Loadings are unavailable"
    print(
        f"""
    Results for {score}:
    Largest loadings: 
    """
    )
    with pd.option_context(
        "display.max_rows",
        5,
        "display.max_columns",
        None,
        "display.width",
        1000,
        "display.precision",
        3,
        "display.colheader_justify",
        "center",
    ):
        print(tops)

print('e')