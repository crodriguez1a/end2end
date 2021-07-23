import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import ast
#import wandb
import math

class RMSLELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        
    def forward(self, pred, actual):
        return self.mse(torch.log(pred.float() + 1), torch.log(actual.float() + 1))

from featureblocks_16khz import FeatureBlock3


frame_length = 16000
overlapping_fraction = 0.1
data = torch.load('./torch_dataset_16khz/all_audio_data.pt')
#print(data.size())

X_train = data[:,0:frame_length].clone()
test_portion = int(0.80*((data.size())[0]))
#print(test_portion)
X_train = data[:test_portion, 0:frame_length].clone()
#print(X_train.size())
Y_train = data[:test_portion,frame_length:].clone()
#print(Y_train.size())

#X_eval = data[test_portion:, 0:frame_length].clone()
#X_eval = X_eval.reshape(-1, 5, 1000)
#Y_eval = data[test_portion:,frame_length:].clone()


#torch.save(X_eval, "./X_eval0716_1.torch")
#torch.save(Y_eval, "./Y_eval0716_1.torch")

Y_train = Y_train.type(torch.LongTensor)
Y_train_one_hot = F.one_hot(Y_train)
#print(Y_train_one_hot)


X_train = X_train.reshape(-1, 16,1000)
print("X_size:{}".format(X_train.size()))
Y_train_one_hot = Y_train_one_hot.reshape(-1,10)
print("Y_size:{}".format(Y_train_one_hot.size()))
audio_dataset = TensorDataset (X_train, Y_train)
audio_dataloader = DataLoader (audio_dataset, batch_size = 250, shuffle= True)


#audio_testset = TensorDataset (X_eval, Y_eval)
#audio_testloader = DataLoader (audio_testset, batch_size = 50, shuffle= True)

# The trinity of models
model = FeatureBlock3().cuda()
#model = FeatureBlockGT()
# This is the losss function
#loss_function = RMSLELoss()
loss_function = torch.nn.CrossEntropyLoss()
# This is what controls the gradient descent
#optimizer = torch.optim.Adadelta(model.parameters(),lr=0.00001)
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.1)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, threshold_mode='rel', cooldown=0,min_lr=0, eps=1e-08, verbose=False)
T_max = 500
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_max)
iteration = 0


'''
wandb.init(project='end2end1D')
config = wandb.config

model.train()
wandb.watch(model)
'''
train_accu = []
train_losses = []

eval_losses=[]
eval_accu=[]

best_loss = math.inf
for epoch in range(1000):
    running_loss=0
    correct=0
    total=0

    print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))
    losses = []
    for index,(x,y) in enumerate(audio_dataloader):
        y_hat = y.cuda()
        optimizer.zero_grad()
        #print(x.size())
        outputs = model(x.float().cuda())
        # Use argmax to get class with max probability value from softmax
        #x = x.argmax(dim=-1) 
        outputs = outputs.float()
        
        y_hat = y_hat.squeeze(1)
        print("y size:{} outputs size:{}".format(y_hat.size(),outputs.size()))
        loss = loss_function(outputs, y_hat)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += y_hat.size(0)
        correct += predicted.eq(y_hat).sum().item()
        
        #print("iteration:{} loss:{} ".format(iteration, loss.item()))
        losses.append(loss.item())
        iteration += 1
        
        scheduler.step()
                
        #wandb.log({"loss": loss, "epoch": epoch, "learning_rate": optimizer.param_groups[0]['lr']})
    
    # calculate avg loss
    avg_loss = sum(losses) / len(losses)
    
    train_loss=running_loss/len(audio_dataloader)
    print(train_loss)
    accu=100.*correct/total
    print(accu)
    
    train_accu.append(accu)
    train_losses.append(train_loss)
    print('Train Loss: %.3f | Accuracy: %.3f'%(train_loss,accu))
    #wandb.log({"Accuracy": accu})


    
    if avg_loss < best_loss:
        print("Found best loss: {}, saving model...".format(avg_loss))
        best_loss = avg_loss
        torch.save(
            {
                "loss": best_loss,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            },
            "./model0716_1.torch"
        )

'''
    running_loss=0
    correct=0
    total=0
    
    for index,(x,y) in enumerate(audio_testloader):

        outputs = x.float()
        y_hat = y.type(torch.LongTensor)
        outputs = outputs.cuda()
        y_hat = y_hat.cuda()
        model = model.cuda()
        outputs = model(outputs)
        y_hat = y_hat.squeeze(1)
        #import pdb; pdb.set_trace()
        loss = loss_function(outputs,y_hat)
        running_loss+=loss.item()

        _, predicted = outputs.max(1)
        total += y_hat.size(0)
        correct += predicted.eq(y_hat).sum().item()

        print("iteration:{} loss:{} ".format(iteration, loss.item()))
        losses.append(loss)
        iteration += 1

    test_loss=running_loss/len(audio_testloader)
    accu=100.*correct/total

    eval_losses.append(test_loss)
    eval_accu.append(accu)

    print('Test Loss: %.3f | Accuracy: %.3f'%(test_loss,accu)) 
    
torch.save(eval_accu, "./eval_accu0716_1.torch")
torch.save(eval_losses, "./eval_losses0716_1.torch")
'''
torch.save(train_accu, "./train_accu0716_1.torch")
torch.save(train_losses, "./train_losses0716_1.torch")