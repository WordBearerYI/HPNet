from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import os
import torch

def one_hot(lst,num_cls):
    tables = np.zeros((len(lst),num_cls))
    for i,line in enumerate(lst):
        #print(line)
        for j in line:
            tables[i,int(j)] = 1

#print(tables[:5],lst[:5])
return tables

class HPAdataset(Dataset):
    def __init__(self,csv_path,img_path,mode_train,num_cls=19):
        info = pd.read_csv(csv_path)
        
        labels = [x.split('|') for x in info['Label']]
        img_names = info['ID']
        
        self.mode_train = mode_train
        self.imgs = list()
        self.labels = one_hot(labels,num_cls)
        
        name_base = os.path.join(img_path,'train' if mode_train else 'test')
        for img_name in img_names:
            img_name = os.path.join(name_base,img_name)
            
            green = Image.open(img_name+'_green.png')
            red = Image.open(img_name+'_red.png')
            yellow = Image.open(img_name+'_yellow.png')
            blue = Image.open(img_name+'_blue.png')
            self.imgs.append([green,red,yellow,blue])
        print(len(self.imgs))
        self.preprocess = transforms.Compose([
                                              transforms.Resize(256),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                              ])
    
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self,idx):
        green,red,yellow,blue = self.imgs[idx]
        
        green = self.preprocess(green)
        red = self.preprocess(red)
        yellow = self.preprocess(red)
        blue = self.preprocess(blue)
        
        label = torch.tensor([self.labels[idx]])
        img = torch.tensor([green,red,yellow,blue]).double()
        
        return img,label
