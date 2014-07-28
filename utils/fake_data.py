import numpy as np
import sys
if __name__=="__main__":
    if len(sys.argv) != 2:
        print "fake_data.py <nb_examples>"
        sys.exit(1)
    size = 48
    channels =3
    pos = np.random.randint(256,size=(int(sys.argv[1]),size*size*channels))
    np.save("/data/lisatmp3/chassang/facedet/fake_pos.npy",pos)
    del pos
    neg = np.random.randint(256,size=(int(sys.argv[1]),size*size*channels))
    np.save("/data/lisatmp3/chassang/facedet/fake_neg.npy",neg)
    del neg
    print "Done"

