from torch import Dataset
import padnas as pd
import utils

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
        in_data = minibatch['seq']
        in_data.apply(lambda x : torch.cat([AAmap[x], torch.zeros()]))
        in_data = torch.array(in_data)
        return in_data
