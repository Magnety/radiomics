import radiomics
import radiomics.featureextractor as FEE
import SimpleITK as sitk
import os
import pandas as pd
import csv
para_path = 'default.yaml'
input_path = r"G:\Breast\Dataset\breast_input\breast_data_153_wclsmask"
output_path = r"G:\Breast\Dataset\breast_input\breast_data_153_wclsmask_features"
patient_names = os.listdir(input_path)
for patient_name in patient_names:
    print(patient_name)
    path = input_path+'/'+patient_name
    img_path = path+'/imaging.nii.gz'
    mask_path = path+'/segmentation.nii.gz'
    label_path = path+'/label.txt'
    source = open(label_path)  # 打开源文件
    indate = source.read()  # 显示所有源文件内容
    extractor = FEE.RadiomicsFeatureExtractor(para_path)
    print("Extraction parameters:\n\t", extractor.settings)
    print("Enabled filters:\n\t", extractor.enabledImagetypes)
    print("Enabled filters:\n\t", extractor.enabledFeatures)
    output_dir = output_path+'/'+patient_name
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
# 运行
    result = extractor.execute(img_path, mask_path,label=int(indate)+1)  # 抽取特征
    print("Result type:", type(result))  # result is returned in a Python ordered dictionary
    print("")
    print("Calculated features")
    for key, value in result.items():  # 输出特征
        print("\t", key, ":", value)
    #pd.DataFrame(result).to_csv(output_dir+'/feature.csv')
    except_keys = ["diagnostics_Versions_PyRadiomics","diagnostics_Versions_Numpy","diagnostics_Versions_SimpleITK","diagnostics_Versions_PyWavelet","diagnostics_Versions_Python",\
                   "diagnostics_Configuration_Settings","diagnostics_Configuration_EnabledImageTypes","diagnostics_Image-original_Hash","diagnostics_Image-original_Dimensionality",\
                   "diagnostics_Image-original_Spacing","diagnostics_Image-original_Size","diagnostics_Mask-original_Hash","diagnostics_Mask-original_Spacing",\
                   "diagnostics_Mask-original_Size","diagnostics_Mask-original_BoundingBox","diagnostics_Mask-original_CenterOfMassIndex","diagnostics_Mask-original_CenterOfMass"]

    with open(output_dir+'/feature.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in result.items():
            if key not in except_keys:
                writer.writerow([key, value])
