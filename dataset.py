from torch.utils.data import Dataset
import pandas as pd 
import numpy as np
import cv2
from PIL import Image
import os
import torch

def one_hot(lst,num_cls):
    tables = np.zeros((len(lst),num_cls))
    for _,line in enumerate(lst):
        tables[line] = 1
    print(tables[:5],lst[:5])
    return tables

class HPAdataset(Dataset):
    def __init__(self,csv_path,img_path,mode_train=True,num_cls=19):
        info = pd.read_csv(csv_path)
        labels = [x.split('|') for x in info['Label']]
        img_names = info['ID']
        
        self.mode_train = mode_train
        self.imgs = list()
        self.labels = one_hot(labels,num_cls)

        name_base = os.path.join(img_path,'train' if mode_train else 'test')
        for img_name in img_names:
            img_name = os.path.join(name_base,img_name)
            
            green = cv2.imread(img_name+'_green.jpg')
            red = cv2.imread(img_name+'_red.jpg')
            yellow = cv2.imread(img_name+'_yellow.jpg')
            blue = cv2.imread(img_name+'_blue.jpg')
            self.imgs.append([green,red,yellow,blue])

            
    def __len__(self):
        return len(self.labels)

    def __getitem__(self,idx):
        label = torch.tensor(self.labels[idx])
        img = torch.tensor(self.imgs[idx])
        return img,label