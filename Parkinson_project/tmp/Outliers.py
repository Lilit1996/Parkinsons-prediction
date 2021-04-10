import pandas as pd

df = pd.read_csv(f"../src/data/parkinsons_data.txt")
df = df.drop(columns=['name'])

features_with_outliers = ["MDVP:Fhi(Hz)", "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
                          "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA",
                          "NHR", "HNR", "D2", "spread1", "spread2", "PPE"]
outlier_indexes = []
for col in features_with_outliers:
    Q1 =  df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outlier_indexes.append(df[(df[col] < (Q1-1.5*IQR)) | (df[col] > (Q3+1.5*IQR))].index.tolist())

flat_list = [item for sublist in outlier_indexes for item in sublist]

outlier_indexes = list(set(flat_list))
print(len(outlier_indexes))

outliers = df[df.index.isin(outlier_indexes)]
print(outliers)