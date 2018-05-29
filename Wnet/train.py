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
            #can be optimized by shrink the A to a one-channel image
            A = torch.zeros(k.size()[1:])
            for i in range(pred.size[1]):
                for j in range(pred.size[2]):
                    most_likehood = 0
                    for k in range(pred.size[0]):
                        if pred[k,i,j] > most_likehood:
                            most_likehood = pred[k,i,j]
                            A[i,j] = k
            
            #Reconstruction Loss
            pred,rec_image = model(x)
            rec_loss = reconstr(x,rec_image)
            optimizer.zero_grad()
            rec_loss.backward()
            optimizer.step()

