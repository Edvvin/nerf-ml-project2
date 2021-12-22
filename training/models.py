import torch
import torch.nn as nn

class AASequenceCNNModel(nn.Module):
    
    def __init__(self, conv_depth=4, lin_depth=2):
        super(AASequenceCNNModel, self).__init__()
        self.conv_depth = conv_depth
        self.lin_depth = lin_depth
        self.conv1 = nn.Conv1d(20,128,20, padding='same')
        self.convHidden = nn.ModuleList([nn.Conv1d(128,128,20, padding='same') 
                                                        for i in range(self.conv_depth)])
        self.dropHidden = nn.ModuleList([nn.Dropout(0.3) for i in range(self.conv_depth)])
        self.reluHidden = nn.ModuleList([nn.ReLU() for i in range(self.conv_depth)])
        self.linLayers = nn.ModuleList([nn.Linear(128, 128) for i in range(self.lin_depth)])
        self.dropLin = nn.ModuleList([nn.Dropout(0.3) for i in range(self.lin_depth)])
        self.reluLin = nn.ModuleList([nn.ReLU() for i in range(self.lin_depth)])
        self.linLast = nn.Linear(128,1)
        self.sig = nn.Sigmoid()
        
    def forward(self, x):
        y = torch.transpose(x, 1, 2)
        y = self.conv1(y.float())
        for i in range(self.conv_depth):
            y = self.dropHidden[i](y)
            y = self.convHidden[i](y)
            y = self.reluHidden[i](y)
        y = torch.transpose(y, 1, 2)
        for i in range(self.lin_depth):
            y = self.dropLin[i](y)
            y = self.linLayers[i](y)
            y = self.reluLin[i](y)
        y = self.linLast(y)
        y = self.sig(y)
        return y
