import torch
import torch.nn as nn
import torch.nn.functional as func

class NCutsLoss(nn.Module):
    def __init__(self):
        super(NCutsLoss,self).__init__()

    def forward(self, seg, dissim, A):
        K,X,Y = seg.size()
        Nassoc = torch.zeros(1)
        for k in range(K):
            assocA = torch.zeros(1)
            for i in range(X):
                for j in range(Y):
                    temp = torch.zeros(1)
                    for m in range(X):
                        for n in range(Y):
                            temp = torch.add(temp, other = torch.mul(dissim[i,j,m,n], seg[k,m,n]))
                    assocA = torch.add(assocA, other = torch.mul(seg[k,i,j], temp))
            assocV = torch.zeros(1)
            for i in range(X):
                for j in range(Y):
                    assocV = torch.add(assocV, other = torch.mul(seg[k,i,j],torch.sum(dissim[i,j])))
