

InputCh=3
ScaleRatio = 2
ConvSize = 3
pad = (ConvSize - 1) / 2 
MaxLv = 5
ChNum = [3,64]
for i in range(MaxLv-1):
    ChNum.append(ChNum[-1]*2)
datapath = "../BSR/BSDS500/data"

