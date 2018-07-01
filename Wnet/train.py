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
cuda_device = torch.cuda.device(config.cuda_dev)
if __name__ == '__main__':
    dataset = DataLoader(config.datapath,"train")
    dataloader = dataset.torch_loader()
    model = WNet()
    model.cuda(config.cuda_dev)
    #optimizer
    optimizer = torch.optim.SGD(model.parameters(),lr = config.init_lr)
    reconstr = torch.nn.MSELoss().cuda(config.cuda_dev)
    Ncuts = NCutsLoss().cuda(config.cuda_dev)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_decay_iter, gamma=config.lr_decay)
    for epoch in range(config.max_iter):
        print("Epoch: "+str(epoch+1))
        scheduler.step()
        
        for step,[x,w] in enumerate(dataloader):
            print('Step' + str(step+1))
            #NCuts Loss
                    
            pred,rec_image = model(x)
            #pred.cuda()
            ncuts_loss = Ncuts(pred,w)
            print("NCuts Loss: " + str(ncuts_loss.item()))
            optimizer.zero_grad()
            ncuts_loss.backward()
            optimizer.step()
        
            #Reconstruction Loss
            pred,rec_image = model(x)
            rec_loss = reconstr(rec_image,x)
            print("Reconstruction Loss: " + str(rec_loss.item()))
            optimizer.zero_grad()
            rec_loss.backward()
            optimizer.step()
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
                    'Ncuts': ncuts_loss.item(),
                    'recon': rec_loss.item()
                    },f)
            print(checkname+' saved')
