import torch
import numpy as np
from configure import Config
from model import WNet
from Ncuts import NCutsLoss
from DataLoader import DataLoader
import time
import os
import torchvision
import pdb
from PIL import Image

config = Config()
cuda_device = torch.cuda.device(config.cuda_dev)
if __name__ == '__main__':
    dataset = DataLoader(config.datapath,"test")
    dataloader = dataset.torch_loader()
    model = WNet()
    
    model.cuda(config.cuda_dev)
    #optimizer
    optimizer = torch.optim.SGD(model.parameters(),lr = config.init_lr)
    reconstr = torch.nn.MSELoss().cuda(config.cuda_dev)
    Ncuts = NCutsLoss().cuda(config.cuda_dev)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_decay_iter, gamma=config.lr_decay)
    with open(config.model_tested,'rb') as f:
        para = torch.load(f)
        model.load_state_dict(para['state_dict'])
        optimizer.load_state_dict(para['optimizer'])
        scheduler.load_state_dict(para['scheduler'])
        
    for step,[x,w] in enumerate(dataloader):
        print('Step' + str(step+1))
            #NCuts Loss
                    
        x = x.cuda(config.cuda_dev)
        w = w.cuda(config.cuda_dev)
        x.requires_grad = False
        w.requires_grad = False
        pred,rec_image = model(x)
        #pdb.set_trace()
        seg = (pred.argmax(dim = 1).to(torch.float)/3*255).cpu().detach().numpy()
        rec_image = rec_image.cpu().detach().numpy()*255
        x = x.cpu().detach().numpy()*255
        pdb.set_trace()
        x = np.transpose(x.astype(np.uint8),(0,2,3,1))
        rec_image = np.transpose(rec_image.astype(np.uint8),(0,2,3,1))
        seg = seg.astype(np.uint8)
        #pdb.set_trace()
        for i in range(seg.shape[0]):
            Image.fromarray(x[i]).save("./input_"+str(step+1)+"_"+str(i)+".jpg")
            Image.fromarray(seg[i]).save("./seg_"+str(step+1)+"_"+str(i)+".jpg")
            Image.fromarray(rec_image[i]).save("./rec_"+str(step+1)+"_"+str(i)+".jpg")



        
