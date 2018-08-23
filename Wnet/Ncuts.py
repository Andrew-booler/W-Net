import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.autograd import Function
import time
import pdb
import subprocess
import numpy as np
from configure import Config

config = Config()

class NCutsLoss(nn.Module):
    def __init__(self):
        super(NCutsLoss,self).__init__()
        self.gpu_list = []
        '''
        for i in range(torch.cuda.device_count()):
            self.gpu_list.append(torch.cuda.device(i))
# the ratio of the free space among all gpus
        self.gpu_room_list = []
        self.gpu_room_update()
        '''
    '''def gpu_room_update(self):
        self.gpu_room_list = []
        free_memory = get_gpu_memory_map()
        total_free = 0
        count_ratio = 0.0
        for _, value in free_memory.items():
            total_free+=value
        for dev in self.gpu_list:
            ratio = float(free_memory[dev])/total_free
            self.gpu_room_list.append(ratio)
            count_ratio += ratio
        if (count_ratio - 1 < 0):
            self.gpu_room_list[-1]+=1.0-count_ratio 
    '''    
            

    def forward(self, seg, padded_seg, weight,sum_weight):
        #too many values to unpack
        cropped_seg = []
        for m in torch.arange((config.radius-1)*2+1,dtype=torch.long):
            column = []
            for n in torch.arange((config.radius-1)*2+1,dtype=torch.long):
                column.append(padded_seg[:,:,m:m+seg.size()[2],n:n+seg.size()[3]].clone())
            cropped_seg.append(torch.stack(column,4))
        cropped_seg = torch.stack(cropped_seg,4)
        #for m in torch.arange(50,70,dtype=torch.long):

        #    print(m)
        #    for n in torch.arange(50,70,dtype= torch.long):
        #        print(weight[5,0,m,n])
        multi1 = cropped_seg.mul(weight)
        multi2 = multi1.sum(-1).sum(-1).mul(seg)
        multi3 = sum_weight.mul(seg)
        #print("=============================================================================")
        #for a in [0,1]:
        #    print(multi2[5,0,a*10+50:a*10+60,50:60])
        #    print(multi2[5,0,a*10+50:a*10+60,60:70])
        assocA = multi2.view(multi2.shape[0],multi2.shape[1],-1).sum(-1)
        assocV = multi3.view(multi3.shape[0],multi3.shape[1],-1).sum(-1)
        assoc = assocA.div(assocV).sum(-1)
        
        return torch.add(-assoc,config.K)
        
    '''def crop_seg(self,seg):
        cropped_seg = torch.zeros(seg.size()[0],seg.size()[1],seg.size()[2],seg.size()[3],(config.radius-1)*2+1,(config.radius-1)*2+1)
        padding_size = (config.radius,config.radius,config.radius,config.radius)
        padded_seg = torch.nn.functional.pad(seg,padding_size)
        for m in torch.arange((config.radius-1)*2+1,dtype=torch.long):
            for n in torch.arange((config.radius-1)*2+1,dtype=torch.long):
                cropped_seg[:,:,:,:,m,n].copy_(padded_seg[:,:,m:m+seg.size()[2],n:n+seg.size()[3]])
        return cropped_seg
    
def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory free as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.free',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map
'''        


