from torchvision.models import resnext50_32x4d as resnext
import torch.nn as nn
import torch
import torch.nn.functional as F


class HPANet(nn.Module):
    def __init__(self, hidden_size=1280,num_cls=18):
        super(HPANet, self).__init__()
        self.model_base = resnext().double()
        #self.ln = nn.LayerNorm((4,)
        self.linear1 = nn.Linear(1000*4,hidden_size)
        self.linear2 = nn.Linear(hidden_size,hidden_size)
        self.linear3 = nn.Linear(hidden_size,num_cls)
        
        self.relu = nn.ReLU()
        self.loss = nn.CrossEntropyLoss()
        self.dropout = nn.Dropout(0.5)
        self.loss = nn.BCEWithLogitsLoss()
    
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
        x = self.dropout(self.relu(self.ln2(x)))
        x = self.ln3(x)
        return self.loss(labels,x)
    
    
    
    def inference(self,inputs):
        '''
            inputs: 4 x [img_size x img_size x num_channel]
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
        x = self.dropout(self.relu(self.ln2(x)))
        x = self.ln3(x)
        preds = [i for i,x in enumerate(x) if x>=0.5]
        return preds

