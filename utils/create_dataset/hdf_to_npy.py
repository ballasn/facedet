import sys
import numpy as np
try:
    import tables
except ImportError:
    warnings.warn("Couldn't import tables, so far DenseFeat is "
                  "only supported with PyTables")


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage %s: <hdf_file> <nb_examples> <npy_file>" % sys.argv[0])
        sys.exit(1)

    h_file = sys.argv[1]
    nb = int(sys.argv[2])
    npy_file = sys.argv[3]

    h5file = tables.openFile(h_file, mode="r")
    dataset = h5file.getNode('/', "denseFeat")

    print 'hdf :', dataset.shape
    arr = dataset[:nb]
    print 'npy :', arr.shape

    h5file.flush()
    h5file.close()
    np.save(npy_file, arr)
    print 'Done'

