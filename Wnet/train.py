import torch
import numpy as np
from configure import Config
from model import WNet
from Ncuts import NCutsLoss
from DataLoader import DataLoader
import time
import os
import pdb

config = Config()
if __name__ == '__main__':
    dataset = DataLoader(config.datapath,"train")
    dataloader = dataset.torch_loader()
    model = torch.nn.DataParallel(WNet(),config.cuda_dev_list)
    model.cuda(config.cuda_dev)
    #optimizer
    optimizer = torch.optim.SGD(model.parameters(),lr = config.init_lr)
    #reconstr = torch.nn.MSELoss().cuda(config.cuda_dev)
    Ncuts = torch.nn.DataParallel(NCutsLoss(),config.cuda_dev_list).cuda(config.cuda_dev)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_decay_iter, gamma=config.lr_decay)
    for epoch in range(config.max_iter):
        print("Epoch: "+str(epoch+1))
        scheduler.step()
        Ave_Ncuts = 0.0
        #Ave_Rec = 0.0
        for step,[x,w] in enumerate(dataloader):
            #NCuts Loss
            x = x.cuda(config.cuda_dev)
            w = w.cuda(config.cuda_dev)
            pred,rec_image = model(x)
            #pred.cuda()
            ncuts_loss = Ncuts(pred,w).sum()
            Ave_Ncuts = (Ave_Ncuts * step + ncuts_loss.item())/(step+1)
            optimizer.zero_grad()
            ncuts_loss.backward()
            optimizer.step()
            #Reconstruction Loss
            '''pred,rec_image = model(x)
            rec_loss = reconstr(rec_image,x)
            Ave_Rec = (Ave_Rec * step + rec_loss.item())/(step+1)
            optimizer.zero_grad()
            rec_loss.backward()
            optimizer.step()'''
        print("Ncuts loss: "+str(Ave_Ncuts))
        #print("Reconstruction loss: "+str(Ave_Rec))
        if (epoch+1)%1000 == 0:
            localtime = time.localtime(time.time())
            checkname = './checkpoints'
            if not os.path.isdir(checkname):
                os.mkdir(checkname)
            checkname+='/checkpoint'
            for i in range(1,5):
                checkname+='_'+str(localtime[i])
            checkname += '_epoch_'+str(epoch+1)
            with open(checkname,'wb') as f:
                torch.save({
                    'epoch': epoch +1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'Ncuts': Ave_Ncuts#,
                    #'recon': Ave_Rec
                    },f)
            print(checkname+' saved')
