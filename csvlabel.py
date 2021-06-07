import os
import pandas as pd
input_path = r"G:\Breast\Dataset\breast_input\breast_data_153_noclsmask"
output_path = r"G:\Breast\Dataset\breast_input\breast_data_153_output_features\labels.csv"
patient_names = os.listdir(input_path)
source = open(input_path + '/' + patient_names[0] + '/label.txt')  # 打开源文件
indate = source.read()  # 显示所有源文件内容
inp = [{'name':patient_names[0],'label':int(indate)}]
df  =pd.DataFrame(inp).to_csv(output_path,index=False)
for patient_name in patient_names[1:]:
    source = open(input_path + '/' + patient_name+ '/label.txt')  # 打开源文件
    indate = source.read()  # 显示所有源文件内容
    inp = [{'name': patient_name, 'label': int(indate)}]
    df = pd.DataFrame(inp).to_csv(output_path, index=False,mode='a+',header=False)
