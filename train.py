import os
import numpy as np
import tqdm

from matplotlib import cm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
from dataset import HPAdataset
from model import HPANet

seed=42
torch.manual_seed(seed)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lr = 5e-5
batch_size = 64
num_epochs = 100
csv_path = "../input/hpa-single-cell-image-classification/train.csv"
img_path = "../input/hpa-single-cell-image-classification/"

#checkpoint_dir =
#if not os.path.isdir(checkpoint_dir):
#    os.mkdir('results')


dataset_train = HPAdataset(csv_path,img_path, mode_train=True)
dataset_test = HPAdataset(csv_path,img_path, mode_train=False)
loader = DataLoader(dataset_train,batch_size=batch_size,shuffle=True)


def train():
    model = HPANet()
    
    #if opt.resume:
    #model, optimizer = load_checkpoint(os.path.join(checkpoint_dir,'model_best'),model,optimizer)
    
    
    num_total_instance = len(dataset_train)
    num_batch =  np.ceil(num_total_instance/batch_size)
    
    optimizer = optim.AdamW([
                             {
                             "params":model.parameters(), "lr":lr,
                             },
                             #{
                             #"params": model.model_base.parameters(), "lr":lr*0.1,
                             #}
                             ]
                            )
        
                            model.to(device)
                            
                            min_loss = float('inf')
                            for epoch in range(num_epochs):
                                training_loss=0.0
                                    
                                    model.train()
                                    for index,(imgs,labels) in enumerate(loader):
                                        imgs = imgs.to(device)
                                        labels = labels.to(device)
                                        loss = model(imgs,labels)
                                        
                                        optimizer.zero_grad()
                                        loss.backward()
                                        optimizer.step()
                                        training_loss += loss.item()
                                        
                                        print("Epoch:[%d|%d], Batch:[%d|%d]  loss: %f , chamfer: %f, l2: %f"%(epoch,epochs,index,num_batch,loss.item()/batch_size))
                                        if index % validate_inteval == 0:
                                            validate(model)
                                                
                                                training_loss_epoch = training_loss/(len(loader)*opt.batch_size)

if training_loss_epoch < min_loss:
    min_loss = training_loss_epoch
        print('New best performance! saving')
#save_name = os.path.join(checkpoint_dir,'model_best')
#save_checkpoint(save_name,model,latent_vecs,optimizer)

#if (epoch+1) % opt.log_interval == 0:

#save_name = os.path.join(checkpoint_dir,'model_routine')
#save_checkpoint(save_name,model,latent_vecs,optimizer)

def validate(model):
    num_corr = 0
    num_total = len(dataset_test)
    for i,(img,label) in enumerate(dataset_test):
        img = img.to(device).unsqueeze(0)
        pred = model.inference(img)
        if pred==label:
            num_corr += 1
    print(num_corr*1.0/num_total)

if __name__=='__main__':
    train()
