
import SimpleITK as sitk
import cv2
from matplotlib import pyplot as plt
import os
import random
def output_center(label,img,name,indate):
    save_dir = 'G:/tuFramework_data_store/Breast_s/'+name
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    a, b, c = [], [], []
    for i in range(label.shape[0]):
        (mean, stddv) = cv2.meanStdDev(label[i, :, :])
        if mean > 0:
            a.append(i)
    for i in range(label.shape[1]):
        (mean, stddv) = cv2.meanStdDev(label[:, i, :])
        if mean > 0:
            b.append(i)
    for i in range(label.shape[2]):
        (mean, stddv) = cv2.meanStdDev(label[:, :, i])
        if mean > 0:
            c.append(i)
    #abc不正常
    a_s,a_e,b_s,b_e,c_s,c_e = 0,0,0,0,0,0
    a_s = a[0]
    a_e = a[len(a)-1]
    b_s = b[0]
    b_e = b[len(b) - 1]
    c_s = c[0]
    c_e = c[len(c) - 1]

    if a_s > 10:
        a_s -= 10
    if a_e < 308:
        a_e += 10
    if b_s > 10:
        b_s -= 10
    if b_e < 585:
        b_e += 10

    if c_s > 10:
        c_s -= 10
    if c_e < 680:
        c_e += 10

    print(label[a_s:a_e, b_s:b_e, c_s:c_e].shape)
    label_new = sitk.GetImageFromArray(
        label[a_s:a_e, b_s:b_e, c_s:c_e])
    img_new = sitk.GetImageFromArray(
        img[a_s:a_e, b_s:b_e, c_s:c_e])
    img_dir = save_dir +"/imaging.nii.gz"
    label_dir = save_dir +"/segmentation.nii.gz"
    sitk.WriteImage(label_new, label_dir)
    sitk.WriteImage(img_new, img_dir)
    target = open(save_dir + '/label.txt', 'w')  # 打开目的文件
    target.write(str(indate))

j=1
name_root = 'G:/tuFramework_data_store/Breast_c'

names = os.listdir(name_root)
for name in names:

    label_data = sitk.ReadImage(name_root + '/' +name+'/segmentation.nii.gz')
    label = sitk.GetArrayFromImage(label_data)
    img_data = sitk.ReadImage(name_root + '/' +name+'/imaging.nii.gz')
    img = sitk.GetArrayFromImage(img_data)
    source = open(name_root + '/' +name+ '/label.txt')  # 打开源文件
    indate = source.read()  # 显示所有源文件内容
    output_center(label,img,name,indate)
    j+=1