import torch
from torch.utils.data import Dataset
import pandas as pd
import utils

def cut_and_pad(x, cutlen):
    temp = AAmap[x][:cutlen] # cut
    padding = maxlen-len(temp)
    return torch.cat([temp, torch.zeros(padding)]), 
        torch.cat([torch.ones(temp.shape[0]), torch.zeros(padding)])

class AASequenceDataset(Dataset):
    
    def __init__(self, tsv_file, maxlen=None):
        self.data = pd.read_csv(tsv_file, sep='\t')
        if(maxlen == None):
            self.maxlen = torch.max(self.data['seq_len'])
        else:
            self.maxlen = maxlen

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        minibatch = data.iloc[idx]
        seqs = []
        masks = []
        labels = []

        for row in minibatch:
            seq, mask = cut_and_pad(row['seq'], maxlen)
            seqs.append(seq)
            masks.append(mask)
            label = row['Position']
            binlabel = torch.zeros(maxlen)
            binlabel[label] = 1.
            labels.append(binlabel)

        seqs = torch.torch(seqs)
        masks = torch.torch(masks)
        labels = torch.torch(labels)
        return seqs, masks, labels
