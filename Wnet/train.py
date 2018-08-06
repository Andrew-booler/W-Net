import torch
import numpy as np
from configure import Config
from model import Net
from Ncuts import NCutsLoss
from DataLoader import DataLoader
import time
import os
import pdb

config = Config()
if __name__ == '__main__':
    dataset = DataLoader(config.datapath,"train")
    dataloader = dataset.torch_loader()
    model = torch.nn.DataParallel(Net(True),config.cuda_dev_list)
    model.cuda(config.cuda_dev)
    model.train()
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
        t_load = 0.0
        t_forward = 0.0
        t_loss = 0.0
        t_backward = 0.0
        t_adjust = 0.0
        t_reset = 0.0
        for step,[x,w,sw] in enumerate(dataloader):
            #NCuts Loss
            tick = time.time()
            x = x.cuda(config.cuda_dev)
            w = w.cuda(config.cuda_dev)
            sw = sw.cuda(config.cuda_dev)
            t_load += time.time()-tick
            tick = time.time()
            pred,pad_pred = model(x)
            t_forward += time.time()-tick
            #pred.cuda()
            tick = time.time()
            ncuts_loss = Ncuts(pred,pad_pred,w,sw).sum()
            t_loss += time.time()-tick
            tick = time.time()
            Ave_Ncuts = (Ave_Ncuts * step + ncuts_loss.item())/(step+1)
            optimizer.zero_grad()
            t_reset += time.time()-tick
            tick = time.time()
            ncuts_loss.backward()
            t_backward += time.time()-tick
            tick = time.time()
            optimizer.step()
            t_adjust += time.time()-tick
            #Reconstruction Loss
            '''pred,rec_image = model(x)
            rec_loss = reconstr(rec_image,x)
            Ave_Rec = (Ave_Rec * step + rec_loss.item())/(step+1)
            optimizer.zero_grad()
            rec_loss.backward()
            optimizer.step()'''
        t_total = t_load+t_reset+t_forward+t_loss+t_backward+t_adjust
        print("Ncuts loss: "+str(Ave_Ncuts)+";total time: "+str(t_total)+";forward: "+str(t_forward/t_total)+";loss: "+str(t_loss/t_total)+";backward: "+str(t_backward/t_total)+";adjust: "+str(t_adjust/t_total)+";reset&load: "+str(t_reset)+"&"+str(t_load))
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
