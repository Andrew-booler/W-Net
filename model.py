import torch
import torch.nn as nn

class WNet(torch.nn.Module):
    def __init__(self):
        super(WNet, self).__init__()
        self.feature1 = []
        self.feature2 = []
        #U-Net1
        self.conv1 = []
        self.relu1 = []
        self.bn1 = []
        self.maxpool1 = []
        self.uconv1 = []
        #module1-5
        for i in range(1,MaxLv+1):
            self.conv1.append(nn.Conv2d(ChNum[i-1],ChNum[i],ConvSize,padding = pad,bias = False))
            self.relu1.append(nn.ReLu())
            self.bn1.append(nn.BatchNorm(ChNum[i],ChNum[i],ConvSize,padding = pad,bias = False))
            self.conv1.append(nn.Conv2d(ChNum[i],ChNum[i],ConvSize,padding = pad,bias = False))
            self.relu1.append(nn.ReLu())
            self.bn1.append(nn.BatchNorm(ChNum[i],ChNum[i],ConvSize,padding = pad,bias = False))
        #module6-9
        for i in range(MaxLv-1,0,-1):
            self.conv1.append(nn.Conv2d(2*ChNum[i],ChNum[i],ConvSize,padding = pad,bias = False))
            self.relu1.append(nn.ReLu())
            self.bn1.append(nn.BatchNorm(ChNum[i],ChNum[i],ConvSize,padding = pad,bias = False))
            self.conv1.append(nn.Conv2d(ChNum[i],ChNum[i],ConvSize,padding = pad,bias = False))
            self.relu1.append(nn.ReLu())
            self.bn1.append(nn.BatchNorm(ChNum[i],ChNum[i],ConvSize,padding = pad,bias = False))
        #module1-4
        for i in range(MaxLv-1):
            self.maxpool1.append(nn.MaxPool2d(ScaleRatio))
        #module5-8
        for i in range(MaxLv,1,-1):
            self.uconv1.append(nn.ConvTranspose2d(ChNum[i],ChNum[i-1],ScaleRatio,ScaleRatio,bias = True)
        self.predconv = nn.Conv2d(ChNum[1],K,1,padding = pad,bias = False)
        self.softmax = nn.Softmax()
        #U-Net2
        self.conv2 = []
        self.relu2 = []
        self.bn2 = []
        self.maxpool2 = []
        self.uconv2 = []
        #module10
        self.conv2.append(nn.Conv2d(K,ChNum[i],ConvSize,padding = pad,bias = False))
        self.relu2.append(nn.ReLu())
        self.bn2.append(nn.BatchNorm(ChNum[i],ChNum[i],ConvSize,padding = pad,bias = False))
        self.conv2.append(nn.Conv2d(ChNum[i],ChNum[i],ConvSize,padding = pad,bias = False))
        self.relu2.append(nn.ReLu())
        self.bn2.append(nn.BatchNorm(ChNum[i],ChNum[i],ConvSize,padding = pad,bias = False))
        #module11-14
        for i in range(2,MaxLv+1):
            self.conv2.append(nn.Conv2d(ChNum[i-1],ChNum[i],ConvSize,padding = pad,bias = False))
            self.relu2.append(nn.ReLu())
            self.bn2.append(nn.BatchNorm(ChNum[i],ChNum[i],ConvSize,padding = pad,bias = False))
            self.conv2.append(nn.Conv2d(ChNum[i],ChNum[i],ConvSize,padding = pad,bias = False))
            self.relu2.append(nn.ReLu())
            self.bn2.append(nn.BatchNorm(ChNum[i],ChNum[i],ConvSize,padding = pad,bias = False))
        #module15-18
        for i in range(MaxLv-1,0,-1):
            self.conv2.append(nn.Conv2d(2*ChNum[i],ChNum[i],ConvSize,padding = pad,bias = False))
            self.relu2.append(nn.ReLu())
            self.bn2.append(nn.BatchNorm(ChNum[i],ChNum[i],ConvSize,padding = pad,bias = False))
            self.conv2.append(nn.Conv2d(ChNum[i],ChNum[i],ConvSize,padding = pad,bias = False))
            self.relu2.append(nn.ReLu())
            self.bn2.append(nn.BatchNorm(ChNum[i],ChNum[i],ConvSize,padding = pad,bias = False))
        #module10-13
        for i in range(MaxLv-1):
            self.maxpool2.append(nn.MaxPool2d(ScaleRatio))
        #module14-17
        for i in range(MaxLv,1,-1):
            self.uconv2.append(nn.ConvTranspose2d(ChNum[i],ChNum[i-1],ScaleRatio,ScaleRatio,bias = True)
        self.reconsconv = nn.Conv2d(ChNum[1],3,ConvSize,padding = pad,bias = False)
    def forward(self,x):
        self.feature1 = [x]
        #U-Net1
        tempf = self.conv1[0](self.feature1[-1])
        tempf = self.relu1[0](tempf)
        tempf = self.bn1[0](tempf)
        tempf = self.conv1[1](tempf)
        tempf = self.relu1[1](tempf)
        self.feature1 = self.feature1.append(self.bn1[1](tempf))
        for i in range(1,MaxLv):
            tempf = self.maxpool1[i-1](self.feature1[-1])
            tempf = self.conv1[2i](tempf)
            tempf = self.relu1[2i](tempf)
            tempf = self.bn1[2i](tempf)
            tempf = self.conv1[2i+1](tempf)
            tempf = self.relu1[2i+1](tempf)
            self.feature1 = self.feature1.append(self.bn1[2i+1](tempf))
        for i in range(MaxLv+1,2*MaxLv):
            tempf = self.uconv1[i-MaxLv-1](self.feature1[-1])
            tempf = self.tensor.cat(feature1[2*MaxLv-i],tempf)
            tempf = self.conv1[2i](tempf)
            tempf = self.relu1[2i](tempf)
            tempf = self.bn1[2i](tempf)
            tempf = self.conv1[2i+1](tempf)
            tempf = self.relu1[2i+1](tempf)
            self.feature1 = self.feature1.append(self.bn1[2i+1](tempf))
        self.feature1[-1] = self.predconv(self.feature[-1])
        self.feature2 = [self.softmax(self.feature1[-1])]
        #U-Net2
        tempf = self.conv2[0](self.feature2[-1])
        tempf = self.relu2[0](tempf)
        tempf = self.bn2[0](tempf)
        tempf = self.conv2[1](tempf)
        tempf = self.relu2[1](tempf)
        self.feature2 = self.feature2.append(self.bn2[1](tempf))
        for i in range(1,MaxLv):
            tempf = self.maxpool2[i-1](self.feature2[-1])
            tempf = self.conv2[2i](tempf)
            tempf = self.relu2[2i](tempf)
            tempf = self.bn2[2i](tempf)
            tempf = self.conv2[2i+1](tempf)
            tempf = self.relu2[2i+1](tempf)
            self.feature2 = self.feature2.append(self.bn2[2i+1](tempf))
        for i in range(MaxLv+1,2*MaxLv):
            tempf = self.uconv2[i-MaxLv-1](self.feature2[-1])
            tempf = self.tensor.cat(feature2[2*MaxLv-i],tempf)
            tempf = self.conv2[2i](tempf)
            tempf = self.relu2[2i](tempf)
            tempf = self.bn2[2i](tempf)
            tempf = self.conv2[2i+1](tempf)
            tempf = self.relu2[2i+1](tempf)
            self.feature2 = self.feature2.append(self.bn2[2i+1](tempf))
        self.feature2[-1] = self.reconsconv(self.feature2[-1])
        return self.feature2[-1]

