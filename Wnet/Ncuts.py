import torch
import torch.nn as nn
import torch.nn.functional as func

class NCutsLoss(nn.Module):
    def __init__(self):
        super(NCutsLoss,self).__init__()

    def forward(self, seg, dissim):
        K,X,Y = seg.size()
        Nassoc = 0
        for k in range(K):
            assocA = 0
            for i in range(X):
                for j in range(Y):
                    if seg[k,i,j] != 0:
                        temp = 0
                        for m in range(X):
                            for n in range(Y):
                                temp += dissim[i,j,m,n])*seg[k,m,n]
                        assocA += seg[k,i,j] * temp
            assocV = 0
            for 
