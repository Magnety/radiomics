
import SimpleITK as sitk
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import random
def output(mask,img,name,label,feature):
    save_dir = 'G:/tuFramework_data_store/Breast_c_f_noclsmask/'+name
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    mask_new_np = np.zeros_like(mask)
    mask_new_np[mask>0]=1
    mask_new = sitk.GetImageFromArray(
        mask_new_np)
    img_new = sitk.GetImageFromArray(
        img)
    img_dir = save_dir +"/imaging.nii.gz"
    mask_dir = save_dir +"/segmentation.nii.gz"
    sitk.WriteImage(mask_new, mask_dir)
    sitk.WriteImage(img_new, img_dir)
    target = open(save_dir + '/label.txt', 'w')  # 打开目的文件
    target.write(str(label))
    np.save(save_dir +"/feature.npy",feature)

j=1
name_root = 'G:/tuFramework_data_store/Breast_c'

names = os.listdir(name_root)
for name in names:
    mask_data = sitk.ReadImage(name_root + '/' +name+'/segmentation.nii.gz')
    mask = sitk.GetArrayFromImage(mask_data)
    img_data = sitk.ReadImage(name_root + '/' +name+'/imaging.nii.gz')
    img = sitk.GetArrayFromImage(img_data)
    label_data = open(name_root + '/' +name+ '/label.txt')  # 打开源文件
    label = label_data.read()  # 显示所有源文件内容
    feature = np.load('G:/tuFramework_data_store/Breast_s' + '/' +name+ '/feature.npy')
    output(mask,img,name,label,feature)
    j+=1