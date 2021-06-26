#Module to process the audio data into frames of equal lengths with a given overlapping percentage
import os
import numpy as np
import librosa
import torch

def make_frames(filename,folder,frame_length,overlapping_fraction):

    print('processing file' + filename)
    class_id = filename.split('-')[1]
    filename = './UrbanSound8K/audio'+'/'+folder + '/'+filename
    data,sample_rate = librosa.load(filename,sr=16000)
    stride = int((1-overlapping_fraction)*frame_length)
    num_frames = int((len(data)-frame_length)/stride)+1
    temp = np.array([data[i*stride:i*stride+frame_length] for i in range(num_frames)])
    
    temp_tensor = torch.from_numpy(temp)
    
    if(len(temp.shape)==2):
        res = np.zeros(shape=(num_frames,frame_length+1),dtype=np.float64)
        res[:temp.shape[0],:temp.shape[1]] = temp
        res[:,frame_length]=np.array([class_id]*num_frames)
        
        res_tensor = torch.from_numpy(res)
        
        return res_tensor

def make_frames_folder(folders,frame_length,overlapping_fraction):
    dataset=torch.Tensor()
    for folder in folders:
        data = []
        print('processing folder {}'.format(folder))
        files = os.listdir('./UrbanSound8K/audio'+'/'+folder)
        for file in files:
            res = make_frames(file,folder,frame_length,overlapping_fraction)
            if res is not None:
                data.append(res)
        
        local_dataset = data[0]
        print('Stacking data'+folder)
        for i in range(1,len(data)):
            #import pdb;pdb.set_trace()
            local_dataset = torch.vstack((local_dataset,data[i]))
        dataset = torch.cat((dataset,local_dataset))
    torch.save(dataset,'./torch_dataset_16khz/all_audio_data.pt')
    return True
    

models = {}
frame_length = 16000 
overlapping_fraction = 0.5
folders = os.listdir('./UrbanSound8K/audio')
wrote_to_file = make_frames_folder(folders,frame_length,overlapping_fraction)
