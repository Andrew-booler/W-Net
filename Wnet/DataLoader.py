from PIL import Image
import torch
import torch.utils.data as Data
import os
import glob
import numpy as np
import pdb
from configure import Config
import math

config = Config()

class DataLoader():
    #initialization
    #datapath : the data folder of bsds500
    #mode : train/test/val
    def __init__(self, datapath,mode):
        #image container
        self.raw_data = []
        #navigate to the image directory
        images_path = os.path.join(datapath,'images')
        train_image_path = os.path.join(images_path, mode)
        train_image_regex = os.path.join(train_image_path, '*.jpg')
        #find all the images
        file_list = glob.glob(train_image_regex)
        #load the images
        for file_name in file_list:
            image = Image.open(file_name)
            if image.mode != "RGB":
                image = image.convert("RGB")
            self.raw_data.append(image)
        #resize and align
        self.scale_to(config.inputsize[0],config.inputsize[1])
        #normalize
        self.normalize()
        
    
    def scale_to(self, width, height):
        for i in range(len(self.raw_data)):
            image = np.array(self.raw_data[i].resize((width,height),Image.BILINEAR))
            self.raw_data[i] = np.stack((image[:,:,0],image[:,:,1],image[:,:,2]),axis = 0)
        self.raw_data = np.stack(self.raw_data,axis = 0)

    def normalize(self):
        #just for RGB 8-bit color
        self.raw_data = self.raw_data.astype(np.float)/256

    def torch_loader(self):
        return Data.DataLoader(
                                dataset = Data.TensorDataset(torch.from_numpy(self.raw_data).float()),
                                batch_size = config.BatchSize,
                                shuffle = config.Shuffle,
                                num_workers = config.LoadThread
				
                            )
#Memory out, depressed
    def cal_dissim(self,raw_data,shape):
        dissim = np.zeros((shape[0],shape[2],shape[3],shape[2],shape[3]))
        for idx in range(shape[0]):
            for i in range(shape[2]):
                for j in range(shape[3]):
                    dissim[idx,i,j,i,j] = 0.0
                    for m in range(i):
                        for n in range(j):
                            dissim[idx,i,j,m,n] = dissimilarity(raw_data[idx,:,i,j],raw_data[idx,:,m,n])
                            dissim[idx,m,n,i,j] = dissim[idx,i,j,m,n]

    def dissimilarity(im1,im2):
        res = 0
        for i in range(im1.shape[0]):
            res += (im1[i]-im2[i])**2
        return math.sqrt(res)

        




