import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import ast
#import wandb

class RMSLELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        
    def forward(self, pred, actual):
        return self.mse(torch.log(pred.float() + 1), torch.log(actual.float() + 1))

from featureblocks import FeatureBlock3


frame_length = 16000
overlapping_fraction = 0.5
data = torch.load('./torch_dataset_16khz/all_audio_data.pt')
#print(data.size())

X_train = data[:,0:frame_length].clone()
test_portion = int(0.8*((data.size())[0]))
#print(test_portion)
X_train = data[:test_portion, 0:frame_length].clone()
#print(X_train.size())
Y_train = data[:test_portion,frame_length:].clone()

X_train = X_train.reshape(-1,16,1000)

#print(X_train.size())
#print(Y_train.size())
#print(Y_train)

Y_train = Y_train.type(torch.LongTensor)
Y_train_one_hot = F.one_hot(Y_train)
#print(Y_train_one_hot)
#print(Y_train)


audio_dataset = TensorDataset (X_train, Y_train)
audio_dataloader = DataLoader (audio_dataset, batch_size = 100, shuffle= True)


# The trinity of models
model = FeatureBlock3()
#model = FeatureBlockGT()
# This is the losss function
#loss_function = RMSLELoss()
loss_function = torch.nn.CrossEntropyLoss()
# This is what controls the gradient descent
#optimizer = torch.optim.Adadelta(model.parameters(),lr=0.01)
optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)
iteration = 0
losses = []

#wandb.init(project='end2end1D')
#config = wandb.config

#wandb.watch(model)
#model.train()

for epoch in range(100):
    
    for index,(x,y) in enumerate(audio_dataloader):
        optimizer.zero_grad()
        #print(x.size())
        x = model(x.float())
        #print(x.size())
        #print(y.size())
        # Use argmax to get class with max probability value from softmax
        #x = x.argmax(dim=-1) 
        x = x.float()
        y = y.squeeze(1)
        
        loss = loss_function(x,y)
        loss.backward()
        optimizer.step()
        print("iteration:{} loss:{} ".format(iteration, loss.item()))
        losses.append(loss)
        iteration += 1
        
        #wandb.log({"loss": loss, "epoch": epoch})
        
        
#using wandb to visualize

