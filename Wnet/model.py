import torch
import torch.nn as nn
from configure import Config
import pdb

config = Config()

class WNet(torch.nn.Module):
    def __init__(self):
        super(WNet, self).__init__()
        self.feature1 = []
        self.feature2 = []
        #U-Net1
        #module1
        self.conv1 = [
        nn.Conv2d(config.ChNum[0],config.ChNum[1],config.ConvSize,padding = config.pad,bias = False),
        nn.Conv2d(config.ChNum[1],config.ChNum[1],config.ConvSize,padding = config.pad,bias = False)]
        self.ReLU1 = [nn.ReLU(),nn.ReLU()]
        self.bn1 = [nn.BatchNorm2d(config.ChNum[1]),nn.BatchNorm2d(config.ChNum[1])]	
        self.maxpool1 = []
        self.uconv1 = []
        #module2-5
        for i in range(2,config.MaxLv+1):
            self.conv1.append(nn.Conv2d(config.ChNum[i-1],config.ChNum[i],1,bias = False))
            self.conv1.append(nn.Conv2d(config.ChNum[i],config.ChNum[i],config.ConvSize,padding = config.pad,groups = config.ChNum[i],bias = False))
            self.ReLU1.append(nn.ReLU())
            self.bn1.append(nn.BatchNorm2d(config.ChNum[i]))
            self.conv1.append(nn.Conv2d(config.ChNum[i],config.ChNum[i],1,bias = False))
            self.conv1.append(nn.Conv2d(config.ChNum[i],config.ChNum[i],config.ConvSize,padding = config.pad,groups = config.ChNum[i],bias = False))
            self.ReLU1.append(nn.ReLU())
            self.bn1.append(nn.BatchNorm2d(config.ChNum[i]))
        #module6-8
        for i in range(config.MaxLv-1,1,-1):
            self.conv1.append(nn.Conv2d(2*config.ChNum[i],config.ChNum[i],1,bias = False))
            self.conv1.append(nn.Conv2d(config.ChNum[i],config.ChNum[i],config.ConvSize,padding = config.pad,groups = config.ChNum[i],bias = False))
            self.ReLU1.append(nn.ReLU())
            self.bn1.append(nn.BatchNorm2d(config.ChNum[i]))
            self.conv1.append(nn.Conv2d(config.ChNum[i],config.ChNum[i],1,bias = False))
            self.conv1.append(nn.Conv2d(config.ChNum[i],config.ChNum[i],config.ConvSize,padding = config.pad,groups = config.ChNum[i],bias = False))
            self.ReLU1.append(nn.ReLU())
            self.bn1.append(nn.BatchNorm2d(config.ChNum[i]))
        #module9
        self.conv1.append(nn.Conv2d(2*config.ChNum[1],config.ChNum[1],config.ConvSize,padding = config.pad,bias = False))
        self.ReLU1.append(nn.ReLU())
        self.bn1.append(nn.BatchNorm2d(config.ChNum[1]))
        self.conv1.append(nn.Conv2d(config.ChNum[1],config.ChNum[1],config.ConvSize,padding = config.pad,bias = False))
        self.ReLU1.append(nn.ReLU())
        self.bn1.append(nn.BatchNorm2d(config.ChNum[1]))
        #module1-4
        for i in range(config.MaxLv-1):
            self.maxpool1.append(nn.MaxPool2d(config.ScaleRatio))
        #module5-8
        for i in range(config.MaxLv,1,-1):
            self.uconv1.append(nn.ConvTranspose2d(config.ChNum[i],config.ChNum[i-1],config.ScaleRatio,config.ScaleRatio,bias = True))
        self.predconv = nn.Conv2d(config.ChNum[1],config.K,1,bias = False)
        self.softmax = nn.Softmax2d()
        self.ReLU1.append(nn.ReLU())
        self.bn1.append(nn.BatchNorm2d(config.K))
        self.conv1 = torch.nn.ModuleList(self.conv1)
        self.ReLU1 = torch.nn.ModuleList(self.ReLU1)
        self.bn1 = torch.nn.ModuleList(self.bn1)	
        self.maxpool1 = torch.nn.ModuleList(self.maxpool1)
        self.uconv1 = torch.nn.ModuleList(self.uconv1)
        #U-Net2
        self.conv2 = []
        self.ReLU2 = []
        self.bn2 = []
        self.maxpool2 = []
        self.uconv2 = []
        #module10
        self.conv2.append(nn.Conv2d(config.K,config.ChNum[1],config.ConvSize,padding = config.pad,bias = False))
        self.ReLU2.append(nn.ReLU())
        self.bn2.append(nn.BatchNorm2d(config.ChNum[1]))
        self.conv2.append(nn.Conv2d(config.ChNum[1],config.ChNum[1],config.ConvSize,padding = config.pad,bias = False))
        self.ReLU2.append(nn.ReLU())
        self.bn2.append(nn.BatchNorm2d(config.ChNum[1]))
        #module11-14
        for i in range(2,config.MaxLv+1):
            self.conv2.append(nn.Conv2d(config.ChNum[i-1],config.ChNum[i],1,bias = False))
            self.conv2.append(nn.Conv2d(config.ChNum[i],config.ChNum[i],config.ConvSize,padding = config.pad,groups = config.ChNum[i],bias = False))
            self.ReLU2.append(nn.ReLU())
            self.bn2.append(nn.BatchNorm2d(config.ChNum[i]))
            self.conv2.append(nn.Conv2d(config.ChNum[i],config.ChNum[i],1,bias = False))
            self.conv2.append(nn.Conv2d(config.ChNum[i],config.ChNum[i],config.ConvSize,padding = config.pad,groups = config.ChNum[i],bias = False))
            self.ReLU2.append(nn.ReLU())
            self.bn2.append(nn.BatchNorm2d(config.ChNum[i]))
        #module15-17
        for i in range(config.MaxLv-1,1,-1):
            self.conv2.append(nn.Conv2d(2*config.ChNum[i],config.ChNum[i],1,bias = False))
            self.conv2.append(nn.Conv2d(config.ChNum[i],config.ChNum[i],config.ConvSize,padding = config.pad,groups = config.ChNum[i],bias = False))
            self.ReLU2.append(nn.ReLU())
            self.bn2.append(nn.BatchNorm2d(config.ChNum[i]))
            self.conv2.append(nn.Conv2d(config.ChNum[i],config.ChNum[i],1,bias = False))
            self.conv2.append(nn.Conv2d(config.ChNum[i],config.ChNum[i],config.ConvSize,padding = config.pad,groups = config.ChNum[i],bias = False))
            self.ReLU2.append(nn.ReLU())
            self.bn2.append(nn.BatchNorm2d(config.ChNum[i]))
        #module18
        self.conv2.append(nn.Conv2d(2*config.ChNum[1],config.ChNum[1],config.ConvSize,padding = config.pad,bias = False))
        self.ReLU2.append(nn.ReLU())
        self.bn2.append(nn.BatchNorm2d(config.ChNum[1]))
        self.conv2.append(nn.Conv2d(config.ChNum[1],config.ChNum[1],config.ConvSize,padding = config.pad,bias = False))
        self.ReLU2.append(nn.ReLU())
        self.bn2.append(nn.BatchNorm2d(config.ChNum[1]))
        #module10-13
        for i in range(config.MaxLv-1):
            self.maxpool2.append(nn.MaxPool2d(config.ScaleRatio))
        #module14-17
        for i in range(config.MaxLv,1,-1):
            self.uconv2.append(nn.ConvTranspose2d(config.ChNum[i],config.ChNum[i-1],config.ScaleRatio,config.ScaleRatio,bias = True))
        self.reconsconv = nn.Conv2d(config.ChNum[1],3,1,bias = True)
        self.ReLU2.append(nn.ReLU())
        self.bn2.append(nn.BatchNorm2d(3))
        self.conv2 = torch.nn.ModuleList(self.conv2)
        self.ReLU2 = torch.nn.ModuleList(self.ReLU2)
        self.bn2 = torch.nn.ModuleList(self.bn2)
        self.maxpool2 = torch.nn.ModuleList(self.maxpool2)
        self.uconv2 = torch.nn.ModuleList(self.uconv2)
    def forward(self,x):
        self.feature1 = [x]
        #U-Net1
        tempf = self.conv1[0](self.feature1[-1])
        tempf = self.ReLU1[0](tempf)
        tempf = self.bn1[0](tempf)
        tempf = self.conv1[1](tempf)
        tempf = self.ReLU1[1](tempf)
        self.feature1.append(self.bn1[1](tempf))
        for i in range(1,config.MaxLv):

            tempf = self.maxpool1[i-1](self.feature1[-1])
            tempf = self.conv1[4*i-2](tempf)
            tempf = self.conv1[4*i-1](tempf)
            tempf = self.ReLU1[2*i](tempf)
            tempf = self.bn1[2*i](tempf)
            tempf = self.conv1[4*i](tempf)
            tempf = self.conv1[4*i+1](tempf)
            tempf = self.ReLU1[2*i+1](tempf)
            self.feature1.append(self.bn1[2*i+1](tempf))
        for i in range(config.MaxLv,2*config.MaxLv-2):

            tempf = self.uconv1[i-config.MaxLv](self.feature1[-1])
            
            tempf = torch.cat((self.feature1[2*config.MaxLv-i-1],tempf),dim=1)
            tempf = self.conv1[4*i-2](tempf)
            tempf = self.conv1[4*i-1](tempf)
            tempf = self.ReLU1[2*i](tempf)
            tempf = self.bn1[2*i](tempf)
            tempf = self.conv1[4*i](tempf)
            tempf = self.conv1[4*i+1](tempf)
            tempf = self.ReLU1[2*i+1](tempf)
            
            self.feature1.append(self.bn1[2*i+1](tempf))
        tempf = self.uconv1[config.MaxLv-2](self.feature1[-1])
            
        tempf = torch.cat((self.feature1[1],tempf),dim=1)

        tempf = self.conv1[-2](tempf)
        tempf = self.ReLU1[4*config.MaxLv-4](tempf)
        tempf = self.bn1[4*config.MaxLv-4](tempf)
        tempf = self.conv1[-1](tempf)
        tempf = self.ReLU1[4*config.MaxLv-3](tempf)
            
        self.feature1.append(self.bn1[4*config.MaxLv-3](tempf))
        tempf = self.predconv(self.feature1[-1])
        tempf = self.ReLU1[-1](tempf)
        self.feature1[-1] = self.bn1[-1](tempf)
        self.feature2 = [self.softmax(self.feature1[-1])]
        #U-Net2
        
        tempf = self.conv2[0](self.feature2[-1])
        tempf = self.ReLU2[0](tempf)
        tempf = self.bn2[0](tempf)
        tempf = self.conv2[1](tempf)
        tempf = self.ReLU2[1](tempf)
        self.feature2.append(self.bn2[1](tempf))

        for i in range(1,config.MaxLv):
            tempf = self.maxpool2[i-1](self.feature2[-1])
            tempf = self.conv2[4*i-2](tempf)
            tempf = self.conv2[4*i-1](tempf)
            tempf = self.ReLU2[2*i](tempf)
            tempf = self.bn2[2*i](tempf)
            tempf = self.conv2[4*i](tempf)
            tempf = self.conv2[4*i+1](tempf)
            tempf = self.ReLU2[2*i+1](tempf)
            
            self.feature2.append(self.bn2[2*i+1](tempf))
        for i in range(config.MaxLv,2*config.MaxLv-2):
            tempf = self.uconv2[i-config.MaxLv](self.feature2[-1])
            tempf = torch.cat((self.feature2[2*config.MaxLv-i-1],tempf),dim=1)
            tempf = self.conv2[4*i-2](tempf)
            tempf = self.conv2[4*i-1](tempf)
            tempf = self.ReLU2[2*i](tempf)
            tempf = self.bn2[2*i](tempf)
            tempf = self.conv2[4*i](tempf)
            tempf = self.conv2[4*i+1](tempf)
            tempf = self.ReLU2[2*i+1](tempf)
            tempf = self.bn2[2*i+1](tempf)            
            self.feature2.append(tempf)
        tempf = self.uconv2[config.MaxLv-2](self.feature2[-1])
        tempf = torch.cat((self.feature2[1],tempf),dim=1)
        tempf = self.conv2[-2](tempf)
        tempf = self.ReLU2[4*config.MaxLv-4](tempf)
        tempf = self.bn2[4*config.MaxLv-4](tempf)
        tempf = self.conv2[-1](tempf)
        tempf = self.ReLU2[4*config.MaxLv-3](tempf)
        tempf = self.bn2[4*config.MaxLv-3](tempf)            
        self.feature2.append(tempf)
        tempf = self.reconsconv(self.feature2[-1])
        tempf = self.ReLU2[-1](tempf)
        self.feature2[-1] = self.bn2[-1](tempf)
        return [self.feature2[0],self.feature2[-1]]

