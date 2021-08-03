import torch
import pandas as pd
import librosa
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
import librosa.display
import matplotlib.pyplot as plt
import tarfile
import torch.nn as nn
import torch.nn.functional as F

df=pd.read_csv("../UrbanSound8K/metadata/UrbanSound8K.csv")
classes=list(df["class"].unique())
paths=dict()

for i in range(len(classes)):
    temp_df=df[df["class"]==classes[i]].reset_index()
    fold=temp_df["fold"].iloc[0]    # The fold of the first audio sample for the specific class
    sample_name=temp_df["slice_file_name"].iloc[0]
    path="../UrbanSound8K/audio/fold{0}/{1}".format(fold, sample_name)
    paths[classes[i]]=path

def extract_mfcc(path):
    audio, sr=librosa.load(path)
    mfccs=librosa.feature.mfcc(audio, sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)
    
features=[]
labels=[]
folds=[]

for i in range(len(df)):
    fold=df["fold"].iloc[i]
    filename=df["slice_file_name"].iloc[i]
    path="../UrbanSound8K/audio/fold{0}/{1}".format(fold, filename)
    mfccs=extract_mfcc(path)
    features.append(mfccs)
    folds.append(fold)
    labels.append(df["classID"].iloc[i])
    
features=torch.tensor(features)
labels=torch.tensor(labels)
folds=torch.tensor(folds)
# Saving the dataset to disk to prevent re-Loading
torch.save(features, "./data/features_mfccs.pt")
torch.save(labels, "./data/labels.pt")
torch.save(folds, "./data/folds.pt")