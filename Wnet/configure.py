
class Config:
    
    def __init__(self):
        #network configure
        self.InputCh=3
        self.ScaleRatio = 2
        self.ConvSize = 3
        self.pad = 1#(self.ConvSize - 1) / 2 
        self.MaxLv = 5
        self.ChNum = [self.InputCh,64]
        for i in range(self.MaxLv-1):
            self.ChNum.append(self.ChNum[-1]*2)
        #data configure
        self.pascal = "../VOC2012/JPEGImages"
        self.bsds = "../BSR/BSDS500/data/images/"
        self.imagelist = "ImageSets/Segmentation/train.txt"
        self.BatchSize = 6
        self.Shuffle = True
        self.LoadThread = 4
        self.inputsize = [224,224]
        #partition configure
        self.K = 64
        #training configure
        self.init_lr = 0.05
        self.lr_decay = 0.1
        self.lr_decay_iter = 1000
        self.max_iter = 50000
        self.cuda_dev = 0 
        self.cuda_dev_list = "0,1"
        self.check_iter = 1000
        #Ncuts Loss configure
        self.radius = 4
        self.sigmaI = 10
        self.sigmaX = 4
        #testing configure
        self.model_tested = "./checkpoint_8_23_13_0_epoch_2000"
        #color library
        self.color_lib = []
        for r in range(0,256,128):
            for g in range(0,256,128):
                for b in range(0,256,128):
                    self.color_lib.append((r,g,b))
