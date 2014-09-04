import tables
import sys
from time import time
import numpy as np
import cv2

if __name__ == '__main__':

    if len(sys.argv) != 4:
        print "Usage : %s <size> <hdf_in> <hdf_out>" % sys.argv[0]
        sys.exit(2)

    new_size = int(sys.argv[1])
    hdf_in = sys.argv[2]
    hdf_out = sys.argv[3]
    old_size = 96
    nb_channels = 3

    # Open an existing hdf5 file
    h5file_in = tables.openFile(hdf_in, mode="r")
    h5array_in = h5file_in.getNode('/', "denseFeat")
    print "Got array from input"
    n = len(h5array_in)
    # Create a new hdf file
    f = tables.openFile(hdf_out, 'w')
    atom = tables.Float32Atom()
    filters = tables.Filters(complib='blosc', complevel=0)
    hdf_shape = (n, new_size ** 2 * nb_channels)
    data = f.createCArray(f.root, 'denseFeat', atom,
                          hdf_shape, filters=filters)
    print "Created output table"
    # Fill it with examples
    t0 = time()
    z = float(new_size) / float(old_size)

    for i in range(n):
        img = h5array_in[i, :]
        img = np.reshape(img, (old_size, old_size, 3))
        img = cv2.resize(img, (new_size, new_size))
        img = np.reshape(img, (new_size * new_size * 3))
        data[i, :] = img
        #cv2.imshow('original', img2.astype(np.uint8))
        #cv2.imshow('new', img3.astype(np.uint8))
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        if i%1000 == 0:
            sys.stdout.write('\r'+str(i))
            sys.stdout.flush()
    t = time()
    dt = t-t0
    print ""
    print dt, "seconds for", n, "examples"
    # Save
    t0 = time()
    f.flush()
    f.close()
    t = time()
    dt = t-t0

    print dt, "seconds to save file"
