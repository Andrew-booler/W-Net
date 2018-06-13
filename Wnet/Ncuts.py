import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.autograd import Function
from configure import Config



import time
import pdb
import gc

config = Config()

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
            xi.requires_grad = False
            #print(torch.__version__)
            segi = seg[idx,:,:,:]
            for k in torch.arange(K,dtype=torch.long):
                assocA = torch.zeros(X,Y).cuda()
                a = torch.cuda.memory_allocated()
                for i in torch.arange(X,dtype=torch.long):

                    for j in torch.arange(Y,dtype=torch.long):
                        #dist1 = (x[i,j]-x).norm(p=2,dim=2).pow(2).mul(seg[k])
                        #print(str(x[i,j].size())+"\t"+str(x.size())+"\t"+str(dist1.size())+"\t"+str(seg[k].size()))
                        assocA1 = torch.zeros(X,Y).cuda()

                        for m in torch.arange(max(0,i-5),min(X,i+5),dtype=torch.long):
                            for n in torch.arange(max(0,j-5),min(Y,j+5),dtype=torch.long):
                                dist = torch.tensor([m-i,n-j],dtype=torch.float).cuda().norm(p=2)
                                w = torch.tensor(1,dtype=torch.float)
                                if dist<config.radius:
                                    w = torch.tensor(0,dtype=torch.float).cuda()
                                else:
                                    dist = dist.pow(2).div_(-config.sigmax)
                                    dissim = (xi[i,j]-xi[m,n]).norm(p=2).div(-config.sigmai)
                                    w = dist*dissim
                                assocA1[m,n] = w*segi[k,m,n]
                        assocA[i,j] = segi[k,i,j]*assocA1.sum()
                                

                        #print(i*j)
                        #pdb.set_trace()

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
        



