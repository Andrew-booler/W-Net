import torch
import configure
import numpy as np
import model

if __name__ == '__main__':
    dataset = DataLoader(datapath,"train")
    dataloader = dataset.torch_loader()
    model = WNet()
    #optimizer
    optimizer = torch.optim.SGD()
    reconstr = torch.nn.MSELoss(size_average = False)
    for epoch in range(Maxepoch):
        print("Epoch: "+str(epoch))
        for step,(x,dissim) in enumerate(dataloader):
            
            #NCuts Loss
            pred,rec_image = model(x)
            
            #Reconstruction Loss
            pred,rec_image = model(x)
            rec_loss = reconstr(x,rec_image)
            optimizer.zero_grad()
            rec_loss.backward()
            optimizer.step()
