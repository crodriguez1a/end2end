import torch
import torch.nn as nn
#from gammatone_init import generate_filters
from torch.nn import functional as F

class FeatureBlock3(nn.Module):
    def __init__(self):
        super(FeatureBlock3, self).__init__()
        self.l1 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=64, #padding=64,
        stride=2)
        self.l2 = nn.ReLU()
        self.l3 = nn.BatchNorm1d(32)
        self.l4 = nn.MaxPool1d(2,stride=2,padding=1)
        self.l5 = nn.ReLU()
        self.l6 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=32,
        stride=2)
        self.l7 = nn.ReLU()
        self.l8 = nn.BatchNorm1d(64)
        self.l9 = nn.MaxPool1d(4,stride=3)#padding=1
        self.l10 = nn.ReLU()
        self.l11 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=16, #padding=16, 
        stride =2)
        self.l12 = nn.ReLU()
        self.l13 = nn.BatchNorm1d(128)
        self.l14 = nn.Conv1d(in_channels=128, out_channels=128, 
        kernel_size=8,#padding=16
        stride=2)
        self.l15 = nn.ReLU()
        self.l16 = nn.BatchNorm1d(128)
        #self.l17 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=4, padding=16)
        #self.l18 = nn.ReLU()
        #self.l19 = nn.BatchNorm1d(256)
        #self.l20 = nn.MaxPool1d(4,stride=4, padding=1)
        #self.l21 = nn.ReLU()
        
        self.fc1 = nn.Linear(128, 64)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(64, 64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(64, 10)
        self.outputs = nn.Softmax(dim=-1)

        
    def forward(self,inputs):
        #import pdb; pdb.set_trace()
        x = self.l1(inputs)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)
        x = self.l11(x)
        x = self.l12(x)
        x = self.l13(x)
        x = self.l14(x)
        x = self.l15(x)
        x = self.l16(x)
        #x = self.l17(x)
        #x = self.l18(x)
        #x = self.l19(x)
        #x = self.l20(x)
        #x = self.l21(x)
        x = x.reshape(-1, 128)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = F.softmax(x, dim=-1)
        return x
    
"""class FeatureBlockGT(nn.Module):
    def __init__(self):
        super(FeatureBlockGT, self).__init__()
        self.l1 = nn.Conv1d(in_channels=64,out_channels=64,kernel_size=512,stride=1,padding=64)
        filters = generate_filters(64,512,16000,100,2)
        filters = filters.reshape(16, 16, -1)
        #import pdb; pdb.set_trace()
        self.l1.weight.data = filters
        #self.l1.bias.data = torch.flatten(filters[0])         
        
    def forward(self,inputs):
        #import pdb; pdb.set_trace()
        with torch.no_grad():
            x = self.l1(inputs)
            
        return x

"""
