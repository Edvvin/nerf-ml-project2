import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from utils import *
import re

AA_CNT = 20

def cut_and_pad(x, cutlen):
    """
    Cuts and and adds padding to tensor

    This function converts sequences (i.e. tensors) of variable length into ones of uniform length. 
    This is done by cutting sequences which are too long or by adding zero padding to shorter
    sequences in such a way that the end sequence has exactly cutlen elements.

    Parameters
    ----------
    x : torch.Tensor
        The variable length sequence
    cutlen : int
        The number of elements the sequence will have after this function is called

    Returns
    -------
    torch.Tensor
        Uniform length tensor constructed from the input tensor
    """
    temp = x[:cutlen] # cut
    padding = cutlen-len(temp)
    return (
            torch.cat([temp, torch.zeros(padding)]), 
            torch.cat([torch.ones(temp.shape[0]), torch.zeros(padding)])
        )

class AASequenceDataset(Dataset):
    """
    Dataset class for the sequence only model
    
    This is the dataset for the model which will only take into consideration the amino acid 
    sequence.
    """

    def __init__(self, tsv_file, maxlen=None, onehot_input=True, multihot_output=True, equal_size=False):

        """
        Parameters
        ----------
        tsv_file : String
            The path to the adequate .tsv file
        maxlen : int
            The length all sequences will be cut off at or padded to. If `None` then it will be set to
            the length of the longest sequences (no cutting)
        onehot_input : boolean
            If set to `True` all of the input sequence elements will be expaneded to a onehot encoding.
            Otherwise, the input sequence will be encoded by ordinal values
        mutlihot_output : boolean
            If set to `True` the labels will be arrays of ones and zeros where ones denote the PTM sites
            Otherwise, the labels will be encoded by ordinal values

        Returns
        -------
        torch.Tensor
            The input sequence
        torch.Tesnor
            The mask where values of 1. denote that the input is in use while 0 means that it is masked
        torch.Tensor
            The label
        """

        self.data = pd.read_csv(tsv_file, sep='\t')
        self.equal_size = equal_size
        self.onehot_input = onehot_input
        self.multihot_output = multihot_output
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

        if(self.onehot_input):
            onehot_seq = np.zeros(seq.shape[0]*AA_CNT)
            onehot_seq[(torch.arange(0,seq.shape[0] * AA_CNT, AA_CNT) + seq).int()] = 1.0
            onehot_seq = onehot_seq.reshape(seq.shape[0], AA_CNT)
            seq = onehot_seq

        label = row['Position']
        label = re.findall(r'\d+', label)
        label = list(map(int, label))

        if(self.multihot_output):
            binlabel = torch.zeros(self.maxlen)
            binlabel[label] = 1.
            label = binlabel

        return seq, mask, label
