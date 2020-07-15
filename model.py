import torch
import torch.nn as nn
import torch.nn.functional as F

# best configuration is 50_100_50 with lr = 0.00002
# but (nfeat,50) (50,nout) with lr = 0.0001 works almost as good
class Net(nn.Module):
    def __init__(self, nfeat, nout):
        super(Net, self).__init__()
        self.W = nn.Linear(nfeat, 50)
        self.W2 = nn.Linear(50, nout)
        #self.W3 = nn.Linear(20, nout)
        #self.W4 = nn.Linear(50, nout)

    def forward(self, x):
        x1 = torch.relu(self.W(x))
        #x2 = torch.relu(self.W2(x1))
        #x3 = torch.relu(self.W3(x2))
        x4 = self.W2(x1)
        return x4

