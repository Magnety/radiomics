import numpy as np
import pandas as pd
import os
input_dir = r"G:\Breast\Dataset\breast_input\breast_data_153_noclsmask"
features_dir = "G:/tuFramework_data_store/Br_output_features/output_features.csv"
features = pd.read_csv(features_dir)
X_cols = list(features.drop(columns='name',axis=1).columns)
print(features)
features_noname = features.drop(columns='name')
df_name = pd.DataFrame(features)
df_noname = pd.DataFrame(features_noname[X_cols])
for i in range(len(df_name)):
    name = df_name.loc[i]['name']
    features_np = np.array(df_noname.loc[i])
    print("name:",name)
    print("features:",features_np.shape)
    np.save(input_dir +name+ '/feature.npy', features_np)

#for patient_name in patient_names:
