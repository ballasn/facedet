from faceDataset_cascade import faceDataset as fd
import numpy as np
from time import time
for e in range(4):
    t0 = time()
    inac =  np.random.randint(0, 450000, 5000 * 10**e)
    inac = np.unique(inac)
    n = len(inac)
    np.save("nac.npy", inac)
    a = fd("/data/lisatmp3/chassang/facedet/16/pos16_700.npy",
    "/data/lisatmp3/chassang/facedet/16/pos16_700.npy", "train",
    inactive_examples="nac.npy")
    t= time()
    print t-t0, "seconds", n, "inactives"
