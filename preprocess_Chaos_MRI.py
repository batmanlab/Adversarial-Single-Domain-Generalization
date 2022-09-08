import os
import json
import numpy as np
import dicom2nifti
import SimpleITK as sitk
import cv2

data_dir = '../Data/CHAOS_Train_Sets/Train_Sets/MR/'

indexes = os.listdir(data_dir)

dict_lists = []

for index in indexes:
    img_dir = os.path.join(data_dir,index, 'T2SPIR/DICOM_anon')

    dicom2nifti.dicom_series_to_nifti(img_dir, os.path.join(data_dir,index, 'T2SPIR/DICOM_anon', index+'img.nii.gz'), reorient_nifti=True)
    img = sitk.ReadImage(os.path.join(data_dir,index, 'T2SPIR/DICOM_anon', index+'img.nii.gz'))

    labels = os.listdir(os.path.join(data_dir,index, 'T2SPIR/Ground'))
    labels.sort()
    label_np = []
    for label in labels:
        if 'png' in label:
            label_np.append(cv2.flip(cv2.rotate(cv2.imread(os.path.join(data_dir,index, 'T2SPIR/Ground',label)),cv2.ROTATE_180),1)[:,:,0]/63)

    label_np = np.stack(label_np, axis=2).transpose([2,0,1])
    nii_label = sitk.GetImageFromArray(label_np)
    nii_label.CopyInformation(img)
    sitk.WriteImage(nii_label, os.path.join(data_dir,index, 'T2SPIR/Ground', index+'label.nii.gz'))

    dict_lists.append({'image': os.path.join(data_dir,index, 'T2SPIR/DICOM_anon', index+'img.nii.gz'),
                       'label': os.path.join(data_dir,index, 'T2SPIR/Ground', index+'label.nii.gz')})


with open('CHAOS_MRI', 'w') as fout:
    json.dump(dict_lists, fout)