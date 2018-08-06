import torch
import numpy as np
from configure import Config
from model import Net
from Ncuts import NCutsLoss
from DataLoader import DataLoader
import time
import os
import torchvision
import pdb
from PIL import Image

config = Config()
if __name__ == '__main__':
    dataset = DataLoader(config.datapath,"test")
    dataloader = dataset.torch_loader()
    model = Net(True)
    model.cuda(config.cuda_dev)
    model.eval()
    optimizer = torch.optim.SGD(model.parameters(),lr = config.init_lr)
    #optimizer
    with open(config.model_tested,'rb') as f:
        para = torch.load(f,"cuda:0")
        model.load_state_dict(para['state_dict'],False)
    for step,[x] in enumerate(dataloader):
        print('Step' + str(step+1))
            #NCuts Loss
                    
        x = x.cuda(config.cuda_dev)
        pred = model(x)
        seg = (pred.argmax(dim = 1)).cpu().detach().numpy()
        x = x.cpu().detach().numpy()*255
        x = np.transpose(x.astype(np.uint8),(0,2,3,1))
        color_map = lambda c: config.color_lib[c]
        cmap = np.vectorize(color_map)
        seg = np.moveaxis(np.array(cmap(seg)),0,-1).astype(np.uint8)
        #pdb.set_trace()
        for i in range(seg.shape[0]):
            Image.fromarray(x[i]).save("./input_"+str(step+1)+"_"+str(i)+".jpg")
            #for j in range(seg.shape[-1]):
            #pdb.set_trace()
            Image.fromarray(seg[i,:,:]).save("./seg_"+str(step+1)+"_"+str(i)+".jpg")



        
