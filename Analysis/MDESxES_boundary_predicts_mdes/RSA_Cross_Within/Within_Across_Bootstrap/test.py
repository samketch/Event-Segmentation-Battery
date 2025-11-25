import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
file = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\MDESxES_boundary_predicts_mdes\RSA_Cross_Within\Within_Across_Bootstrap\PCA_1\Bootstrap\global_Mean_PCA_1_within_across_bootstrap.csv"
df = pd.read_csv(file)

plt.hist(df["n_pairs"], bins=20)
plt.title("n_pairs distribution")

plt.figure()
plt.hist(df["z"], bins=30)
plt.title("Z distribution")
plt.show()

print(df["p"].describe())
