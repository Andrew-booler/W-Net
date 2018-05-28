import torch
import configure
import numpy as np

if __name__ == '__main__':
    dataloader = DataLoader(datapath,"train")
    for epoch in range(Maxepoch):
        for step,(x) in enumerate(dataloader):

