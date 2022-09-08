import os
import json


data_dir = '../Data/Cardiac/c0t2lge/'
label_dir = '../Data/Cardiac/c0gt/'

dict_lists = []

for i in range(45):
    dict_lists.append({'image':os.path.join(data_dir,'patient'+str(i+1)+'_C0.nii.gz'),
                           'label':os.path.join(label_dir,'patient'+str(i+1)+'_C0_manual.nii.gz')})

with open('data_lists/Cardiac_C', 'w') as fout:
    json.dump(dict_lists, fout)


data_dir = '../Data/Cardiac/c0t2lge/'
label_dir = '../Data/Cardiac/lgegt/'

dict_lists = []

for i in range(45):
    dict_lists.append({'image':os.path.join(data_dir,'patient'+str(i+1)+'_LGE.nii.gz'),
                           'label':os.path.join(label_dir,'patient'+str(i+1)+'_LGE_manual.nii.gz')})

with open('data_lists/Cardiac_LGE', 'w') as fout:
    json.dump(dict_lists, fout)
