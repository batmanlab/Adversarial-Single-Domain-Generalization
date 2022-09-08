import os
import json


data_dir = '../Data/CT_abdominal/Training/img/'
label_dir = '../Data/CT_abdominal/Training/label/'
data_list = os.listdir(data_dir)

dict_lists = []

for img in data_list:
    dict_lists.append({'image':os.path.join(data_dir,img),
                           'label':os.path.join(label_dir,'label'+img.strip('img'))})

with open('Abdominal_CT', 'w') as fout:
    json.dump(dict_lists, fout)
