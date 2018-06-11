import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.autograd import Function
import time
import pdb
class NCutsLoss(nn.Module):
    def __init__(self):
        super(NCutsLoss,self).__init__()

    def forward(self, seg, x):
#too many values to unpack
        N = seg.size()[0]
        K = torch.tensor(seg.size()[1])
        X = seg.size()[2]
        Y = seg.size()[3]
        Kconst = K.float()
        Kconst = Kconst.cuda()
        
        Nassoc = torch.zeros(K.int().item())
        Nassoc = Nassoc.cuda()
        loss = torch.zeros(1)
        loss = loss.cuda()
        #pdb.set_trace()
        
        print("calculating loss")
        for idx in torch.arange(N,dtype=torch.long):
            print("loss: "+str(idx))

            xi = x[idx,:,:,:].transpose(0,2).transpose(0,1)
            #print(torch.__version__)
            segi = seg[idx,:,:,:]
            for k in torch.arange(K,dtype=torch.long):
                assocA = torch.zeros(X,Y).cuda()
                a = torch.cuda.memory_allocated()
                for i in torch.arange(X,dtype=torch.long):

                    for j in torch.arange(Y,dtype=torch.long):
                        #dist1 = (x[i,j]-x).norm(p=2,dim=2).pow(2).mul(seg[k])
                        #print(str(x[i,j].size())+"\t"+str(x.size())+"\t"+str(dist1.size())+"\t"+str(seg[k].size()))
                        a1 = torch.cuda.memory_allocated()
                        print("mark")
                        print(a1 - a)
                        dist1 = xi.add_(-1,xi[i,j]).norm(p=2,dim=2).pow_(2).mul_(segi[k]).sum().mul_(segi[k,i,j])

                        a3 = torch.cuda.memory_allocated()
                        print(a3-a1)
                        #print(i*j)
                        #pdb.set_trace()
                        assocA[i,j] = dist1
                        a4 = torch.cuda.memory_allocated()
                        print(a4-a3)
                        del dist1
                        a5 = torch.cuda.memory_allocated()
                        print(a5-a4)
                assocV = torch.zeros(X,Y).cuda()

                for i in torch.arange(X,dtype=torch.long):
                    for j in torch.arange(Y,dtype=torch.long):
                        temp = (xi[i,j]-xi).norm(p=2,dim=2).pow(2)
                        assocV[i,j] = torch.mul(segi[k,i,j],temp.sum()).cuda()
                Nassoc[k] = assocA.sum().div(assocV.sum()).cuda()
            print(Kconst)
            print(Nassoc)
            loss += Kconst - Nassoc.sum()
        return loss

    def mydist(x1,x2):
        #Returns the 2-norm by 2 of x1 and x2
        kp_dim = x1.size().com
        dist = (x1-x2)*(x1-x2)
        



