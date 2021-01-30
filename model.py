from torchvision.models import resnext50_32x4d as resnext
import torch.nn as nn
import torch
import torch.nn.functional as F


class HPANet(nn.Module): 
    def __init__(self, hidden_size,num_cls=18):
        super(HPANet, self).__init__()
        self.model_base = resnext()
        
        self.ln1 = nn.LayerNorm(nn.Linear(512,hidden_size))
        self.ln2 = nn.LayerNorm(nn.Linear(hidden_size,num_cls))

        self.relu = nn.ReLU()
        self.loss = nn.CrossEntropyLoss() 
        self.dropout = nn.Dropout(0.5)
    
    def forward(self,inputs,labels):
        '''
        inputs: B x 4 x [img_size x img_size x num_channel]
        labels: B x num_cls
        '''
        
        g = inputs[:,0,:,:]
        r = inputs[:,1,:,:]
        y = inputs[:,2,:,:] 
        b = inputs[:,3,:,:]
        fg = self.model_base(g)
        fr = self.model_base(r)
        fy = self.model_base(y)
        fb = self.model_base(b)

        f = torch.cat((fg,fr,fy,fb),dim=1)
        x = self.dropout(self.relu(self.ln1(f)))
        x = self.ln2(x)
