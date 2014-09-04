import tables
import sys
from time import time
import numpy as np
import cv2


def remove_inactives(hdf_in, hdf_out, inac):

    # Open an existing hdf5 file
    h5file_in = tables.openFile(hdf_in, mode="r")
    h5array_in = h5file_in.getNode('/', "denseFeat")
    print "Got array from input"

    n = len(h5array_in)
    size = h5array_in.shape[1]

    # Create a new hdf file
    f = tables.openFile(hdf_out, 'w')
    atom = tables.Float32Atom()
    filters = tables.Filters(complib='blosc', complevel=0)
    c = 0
    for e in inac:
        if e < n:
            c+=1

    hdf_shape = (n - c, size)
    data = f.createCArray(f.root, 'denseFeat', atom,
                          hdf_shape, filters=filters)
    print "Created output table"

    # Fill it with examples
    t0 = time()
    cur = 0
    cur_inac = 0

    for i in range(n):
        if i == inac[cur_inac]:
            cur_inac += 1
        else:
            data[cur, :] = h5array_in[i, :]
            cur += 1
            if i%1000 == 0:
                sys.stdout.write('\r'+str(i)+"  "+str(cur))
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


if __name__ == '__main__':

    if len(sys.argv) != 4:
        print "Usage : %s <hdf_in> <hdf_out> <inactives>" % sys.argv[0]
        sys.exit(2)

    hdf_in = sys.argv[1]
    hdf_out = sys.argv[2]
    inac = np.load(sys.argv[3])
    print "Got", len(inac), "inactives"
    remove_inactives(hdf_in, hdf_out, inac)
