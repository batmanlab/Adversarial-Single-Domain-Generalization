import os
import json


data_dir = '../Data/Prostate_6/'
folder_lists = ['BIDMC', 'BMC', 'HK', 'I2CVB', 'RUNMC', 'UCL']

for folder in folder_lists:
    dict_lists = []

    data_lists = os.listdir(os.path.join(data_dir, folder))
    data_names = []
    for data in data_lists:
        data_names.append(data.strip('_Segmentation.nii.gz').strip('_segmentation.nii.gz').strip('.nii.gz'))

    data_names = list(set(data_names))

    for data_name in data_names:
        if (data_name+'_Segmentation.nii.gz') in data_lists:
            dict_lists.append({'image': os.path.join(data_dir, folder, data_name+'.nii.gz'),
                           'label': os.path.join(data_dir, folder, data_name+'_Segmentation.nii.gz')})
        elif (data_name+'_segmentation.nii.gz') in data_lists:
            dict_lists.append({'image': os.path.join(data_dir, folder, data_name + '.nii.gz'),
                               'label': os.path.join(data_dir, folder, data_name + '_segmentation.nii.gz')})

    with open(os.path.join('data_lists', folder), 'w') as fout:
        json.dump(dict_lists, fout)
