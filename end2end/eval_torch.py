import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import ast
import wandb
from os import path
import math
import sys

class RMSLELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        
    def forward(self, pred, actual):
        return self.mse(torch.log(pred.float() + 1), torch.log(actual.float() + 1))

from featureblocks_16khz import FeatureBlock3

model_name = "model_final.torch"
frame_length = 16000
overlapping_fraction = 0.1
data = torch.load('./torch_dataset_16khz/all_audio_data.pt')


# Shuffle dataset before we do anything
data=data[torch.randperm(data.size()[0])]


X_train = data[:,0:frame_length].clone()
test_portion = int(0.80*((data.size())[0]))

X_eval = data[test_portion:, 0:frame_length].clone()
X_eval = X_eval.reshape(-1, 16, 1000)
Y_eval = data[test_portion:,frame_length:].clone()


audio_testset = TensorDataset (X_eval, Y_eval)
audio_testloader = DataLoader (audio_testset, batch_size = 50, shuffle= True)


# The trinity of models
model = FeatureBlock3().cuda()
loss_function = torch.nn.CrossEntropyLoss()
# This is what controls the gradient descent
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
T_max = 500
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_max)
iteration = 0
#
#wandb.init(project='end2end1D')
#config = wandb.config


# Load model if exists
if path.exists(model_name):
    checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage.cuda(1))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    best_loss = checkpoint['loss']
    print("Loaded model from {} with loss {}".format(model_name, checkpoint['loss']))

model.eval()
#wandb.watch(model)

train_accu = []
train_losses = []

eval_losses=[]
eval_accu=[]



running_loss=0
correct=0
total=0
losses = []

with torch.no_grad():

    for index,(x,y) in enumerate(audio_testloader):

        outputs = x.float()
        y_hat = y.type(torch.LongTensor)
        outputs = outputs.cuda()
        y_hat = y_hat.cuda()

        model.eval()
        outputs = model(outputs)
        y_hat = y_hat.squeeze(1)
        #print("y size:{} outputs size:{}".format(y_hat.size(),outputs.size()))
        loss = loss_function(outputs, y_hat)

        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += y_hat.size(0)
        correct += predicted.eq(y_hat).sum().item()

        print("iteration:{} loss:{} correct: {} / {} ".format(iteration, loss.item(), correct, total))
        losses.append(loss)
        iteration += 1

test_loss=running_loss/iteration
accu=100.*correct/total

eval_losses.append(test_loss)
eval_accu.append(accu)

print('Test Loss: %.3f | Accuracy: %.3f'%(test_loss,accu)) 
    
torch.save(eval_accu, "./eval_accu0716_1.torch")
torch.save(eval_losses, "./eval_losses0716_1.torch")

torch.save(train_accu, "./train_accu0716_1.torch")
torch.save(train_losses, "./train_losses0716_1.torch")