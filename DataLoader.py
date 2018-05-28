from PIL import Image
import torch
import torch.utils.data as Data
import os
import glob
import numpy as np
import pdb

class DataLoader():
    #initialization
    #datapath : the data folder of bsds500
    #mode : train/test/val
    def __init__(self, datapath,mode):
        #image container
        self.raw_data = []
        #go to the image directory
        images_path = os.path.join(datapath,'images')
        train_image_path = os.path.join(images_path, model)
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
        self.scale_to(224,224)    
    
    def scale_to(self, width, height):
        for i in range(len(self.raw_data)):
            image = np.array(self.raw_data[i].resize((width,height),Image.BILINEAR))
            self.raw_data[i] = np.stack((image[:,:,0],image[:,:,1],image[:,:,2]),axis = 0)
        self.raw_data = np.stack(self.raw_data,axis = 0)

    def torch_loader():
        
loader = DataLoader("/home/andrew/Wnet/BSR/BSDS500/data","train")



