import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from utils import *
import re


def cut_and_pad(x, cutlen):
    temp = torch.tensor(x[:cutlen]) # cut
    padding = cutlen-len(temp)
    return (
            torch.cat([temp, torch.zeros(padding)]), 
            torch.cat([torch.ones(temp.shape[0]), torch.zeros(padding)])
        )

class AASequenceDataset(Dataset):
    
    def __init__(self, tsv_file, maxlen=None):
        self.data = pd.read_csv(tsv_file, sep='\t')
        if(maxlen == None):
            self.maxlen = np.max(self.data['seq_len'])
        else:
            self.maxlen = maxlen

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        seq = row['seq']
        seq = re.findall(r'\d+', seq)
        seq = list(map(int, seq))
        seq, mask = cut_and_pad(seq, self.maxlen)
        label = row['Position']
        binlabel = torch.zeros(self.maxlen)
        label = re.findall(r'\d+', label)
        label = list(map(int, label))
        binlabel[label] = 1.

        return seq, mask, binlabel
