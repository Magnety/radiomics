
import SimpleITK as sitk
import cv2
from matplotlib import pyplot as plt
import os
import random


j=1
name_root = 'G:/tuFramework_data_store/Breast_s'

names = os.listdir(name_root)
for name in names:

    label_data = sitk.ReadImage(name_root + '/' +name+'/segmentation.nii.gz')
    label = sitk.GetArrayFromImage(label_data)
    print("Name:",name)
    print("Shape:",label.shape)
    j+=1


    #25   79    49