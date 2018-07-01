
class Config:
    
    def __init__(self):
        #network configure
        self.InputCh=3
        self.ScaleRatio = 2
        self.ConvSize = 3
        self.pad = (self.ConvSize - 1) / 2 
        self.MaxLv = 5
        self.ChNum = [self.InputCh,64]
        for i in range(self.MaxLv-1):
            self.ChNum.append(self.ChNum[-1]*2)
        #data configure
        self.datapath = "../BSR/BSDS500/data"
        self.BatchSize = 2
        self.Shuffle = False
        self.LoadThread = 2
        self.inputsize = [224,224]
        #partition configure
        self.K = 3
        #training configure
        self.init_lr = 0.003
        self.lr_decay = 0.1
        self.lr_decay_iter = 1000
        self.max_iter = 50000
        #Ncuts loss configure
        self.radius = 5
        self.sigmax = 4**2
        self.sigmai = 10**2
