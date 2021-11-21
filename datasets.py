import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from utils import *
import re

AA_CNT = 20

def cut_and_pad(x, cutlen):
    temp = x[:cutlen] # cut
    padding = cutlen-len(temp)
    return (
            torch.cat([temp, torch.zeros(padding)]), 
            torch.cat([torch.ones(temp.shape[0]), torch.zeros(padding)])
        )

class AASequenceDataset(Dataset):
    
    def __init__(self, tsv_file, maxlen=None, equal_size=False):
        self.data = pd.read_csv(tsv_file, sep='\t')
        self.equal_size = equal_size
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
        seq = torch.tensor(seq)
        if(self.equal_size):
            seq, mask = cut_and_pad(seq, self.maxlen)
        else:
            mask = torch.ones(seq.shape[0])
        onehot_seq = np.zeros(self.maxlen)
        onehot_seq[arange(0,seq.shape[0] * AA_CNT, AA_CNT) + seq] = 1.0
        onehot_seq.reshape(seq.shape[0], AA_CNT)
        label = row['Position']
        binlabel = torch.zeros(self.maxlen)
        label = re.findall(r'\d+', label)
        label = list(map(int, label))
        binlabel[label] = 1.

        return onehot_seq, mask, binlabel
