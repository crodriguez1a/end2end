#NoteBook

import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from model import EnvNet
from train import train_model
from data_preprocess import make_frames,make_frames_folder
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Conv1D,MaxPooling1D,Dense,Flatten,BatchNormalization,Dropout, Activation
from gammatone_init import GammatoneInit
from gammatone_init import generate_filters
from model_config import *
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import ast


frame_length = 16000
overlapping_fraction = 0.5
data = torch.load('./torch_dataset_16khz/all_audio_data.pt')
print(data.size())


def to_categorical(tensors, num_classes=10):
    return torch.eye(num_classes)[y.int()]
print(data.size())
tensor_size = (data.size())[0]
print(tensor_size)



X_train = data[:,0:frame_length].clone()
test_portion = int(0.8*((data.size())[0]))
print(test_portion)
X_train = data[:test_portion, 0:frame_length].clone()
print(X_train.size())
Y_train = data[:test_portion,frame_length:].clone()

X_train = X_train.reshape(-1,16,1000)

print(X_train.size())
print(Y_train.size())
print(Y_train)



audio_dataset = TensorDataset (X_train, Y_train)
audio_dataloader = DataLoader (audio_dataset, batch_size = 100, shuffle= True)
print(Y_train)


sample_rate = 16000
min_center_freq = 100
order = 2


class RMSLELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        
    def forward(self, pred, actual):
        return self.mse(torch.log(pred.float() + 1), torch.log(actual.float() + 1))



from feature_blocks import FeatureBlock3
from feature_blocks import FeatureBlockGT

# The trinity of models
model = FeatureBlock3()
#model = FeatureBlockGT()
# This is the losss function
loss_function = RMSLELoss()
# This is what controls the gradient descent
optimizer = torch.optim.Adadelta(model.parameters(), lr=0.5)
iteration = 0
losses = []

for epoch in range(100):
    
    for index,(x,y) in enumerate(audio_dataloader):
        optimizer.zero_grad()
        # Run model and use .cuda() to send to GPU
        #print(x.size())
        x = model(x.float()).cuda()
        #print(x.size())
        #print(y.size())
        # Use argmax to get class with max probability value from softmax
        x = x.argmax(dim=-1)
        # send y to gpu too
        y = y.cuda()
        
        loss = loss_function(x,y)
        loss.requires_grad = True
        loss.backward()
        optimizer.step()
        print("iteration:{} loss:{} ".format(iteration, loss.item()))
        losses.append(loss)
        iteration += 1


shape = [16,16,64]
filters = generate_filters(shape[2],shape[0],sample_rate,min_center_freq,order)
filters = filters.reshape(filters.shape[1],1,filters.shape[0])


import matplotlib.pyplot as plt
y = np.array(losses, dtype=float)
x = np.arange(len(losses))
print(x.dtype)
print(y.dtype)

m, b = np.polyfit(x, y, 1)
plt.plot(x, y, 'o')
plt.plot(x, m*x+b)