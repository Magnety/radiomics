import os
import pandas as pd
input_path = r"G:\Breast\Dataset\breast_input\breast_data_153_wclsmask_features"
output_path = r"G:\Breast\Dataset\breast_input\breast_data_153_output_features\features.csv"
patient_names = os.listdir(input_path)
dt = pd.read_csv(input_path+'/'+patient_names[0]+'/feature.csv',header=None,index_col=0).T
print(dt)
dt['name'] = patient_names[0]
dt.to_csv(output_path,index=False)
for patient_name in patient_names[1:]:
    #file = pd.read_csv('G:/tuFramework_data_store/Breast_features/case_00016/feature.csv',header=None)
    dt = pd.read_csv(input_path+'/'+patient_name+'/feature.csv',header=None,index_col=0).T
    print(dt)
    dt['name'] = patient_name
    dt.to_csv(output_path,index=False,mode='a+',header=None)