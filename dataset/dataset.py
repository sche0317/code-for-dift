from torch.utils.data import Dataset
import torch
import torch.nn
import os
from torchvision import transforms
import pickle
import torchio as tio
import torch.nn.functional as F
from torchvision.transforms import Normalize, Compose

def load_dict(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

dict = load_dict('subject_list')
'''
note: the dict is a list of dictionaries with the following keys:
dict_keys(['subject_id', 'T1_bet', 'T2_bet', 'PD_bet'])
the values for these keys are the paths to the images
'''

class IXIDataset(torch.utils.data.Dataset):
    def __init__(self, is_train=True, dict_list=dict):
        super().__init__()                               
        self.is_train = is_train
        self.dict = dict_list
        self.subject = None
        self.p2d = (21,21,21,21)

    def __getitem__(self, x):

        self.subject = self.dict[x]

        # for T1_bet
        img_t1 = tio.ScalarImage(self.subject['T1_bet']).data.squeeze(0).permute(2, 0, 1)
        # print(img_t1.shape)
        img_t1 = img_t1[71:111, :, 18:-18].type(torch.float)
        img_t1 = F.pad(img_t1, self.p2d, "constant", 0)

        # for T2_bet
        img_t2 = tio.ScalarImage(self.subject['T2_bet']).data.squeeze(0).permute(2, 0, 1)
        img_t2 = img_t2[71:111, :, 18:-18].type(torch.float) 
        img_t2 = F.pad(img_t2, self.p2d, "constant", 0)

        # for PD_bet
        img_pd = tio.ScalarImage(self.subject['PD_bet']).data.squeeze(0).permute(2, 0, 1)
        img_pd = img_pd[71:111, :, 18:-18].type(torch.float)
        img_pd = F.pad(img_pd, self.p2d, "constant", 0)
            
        # print(img_t1.shape)
        data_dict = {
            "t1" : img_t1,
            "t2" : img_t2,
            "pd" : img_pd
        }
        return data_dict

    def __len__(self):
        return len(self.dict)

        