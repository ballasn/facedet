import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt
### Histogram equalization ###
def histeq(data_file, output_file):
    # Load data
    data  = np.load(data_file)
    for e in xrange(data.shape[0]):
        if e%10==0:
            sys.stdout.write("\r"+str(e))
            sys.stdout.flush()
        img = np.asarray(data[e], dtype = np.uint8)
        img = np.reshape(img, (48,48,3))

        # hist eq
        for i in range(3):
            hist,bins = np.histogram(img[:,:,i].flatten(),256,[0,256])
            cdf = hist.cumsum()
            cdf_normalized = cdf * hist.max()/ cdf.max()
            cdf_m = np.ma.masked_equal(cdf,0)
            cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
            cdf = np.ma.filled(cdf_m,0).astype('uint8')
            img[:,:,i] = np.asarray(cdf[img[:,:,i]], dtype=np.uint8)
        img = img.flatten()
        data[e] = img

    # Assuming [b 0 1 c] axes
    np.save(output_file,data)



if __name__ == '__main__':

    if len(sys.argv) != 3:
        print("Usage %s: <data_dir> <output_dir>" % sys.argv[0])
        exit(1)

    data_dir = sys.argv[1]
    output_dir = sys.argv[2]
    histeq(data_dir, output_dir)
