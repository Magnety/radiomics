import SimpleITK as sitk
import numpy as np
import os


def noclsmask(label_data,label_np,label_np_new,name,indate):
    save_dir = "G:/Breast/Dataset/breast_input/label_nocls"
    #label_np_new = np.zeros_like(label_np)
    label_np_new[label_np>0]=1
    label_new = sitk.GetImageFromArray(label_np_new)
    label_new.SetOrigin(label_data.GetOrigin())
    label_new.SetSpacing(label_data.GetSpacing())
    sitk.WriteImage(label_new,save_dir+'/'+name)
dir = r"G:\Breast\Dataset\breast_input\label"

label_root = r"G:\Breast\Dataset\breast_input\label"
clslabel_root = r"G:\Breast\Dataset\breast_input\class_label"
label_names = os.listdir(label_root)
j=0
for name in label_names:
    label_data = sitk.ReadImage(label_root + '/' + name)
    label_np = sitk.GetArrayFromImage(label_data)
    label_np_new = sitk.GetArrayFromImage(label_data)

    print(name.split('.')[0])
    source = open(clslabel_root + '/' + name.split('.')[0] + '/label.txt')  # 打开源文件
    indate = source.read()  # 显示所有源文件内容
    noclsmask(label_data,label_np,label_np_new,name,int(indate))
    j+=1