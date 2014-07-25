import numpy as np
from math import isnan
import sys


def countNaN(data_file):
    # Load data
    data = np.load(data_file)
    d = data.flatten()
    c = 0
    for e in d:
        if isnan(e):
            c += 1
    return c

##########################
if __name__ == '__main__':

    if len(sys.argv) != 2:
        print("Usage %s: <data_dir>" % sys.argv[0])
        exit(1)

    print countNaN(sys.argv[1])
