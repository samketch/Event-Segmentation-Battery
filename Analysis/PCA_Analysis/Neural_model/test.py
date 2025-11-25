import pandas as pd
df = pd.read_csv(r"C:\Users\Smallwood Lab\Downloads\skipper_data_moviemanifold_5factor_gradient_Common_NativePCAs (2).csv")
print(df.columns.tolist())
print(df['idno'].unique()[:10])
print(df['Task_name'].unique())
