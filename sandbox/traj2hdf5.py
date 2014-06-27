import os
import sys
import numpy
import tables
import math
import gzip


def convert(filelist, indir, outdir):

    buf1 = 0
    with open(filelist) as filelist_fp:
        for line in filelist_fp:
            # Get read/write path
            path1 = indir  + '/' + os.path.splitext(line)[0] + '.traj.gz'
            path2 = outdir + '/' + os.path.splitext(line)[0] + '.hdf'

            print(path1)
            print(path2)
            # Store "x" in a chunked array with level 5 BLOSC compression...
            f = tables.openFile(path2, 'w')
            atom = tables.Float32Atom()
            filters = tables.Filters(complib='blosc', complevel=5)
            dataToInsert1 = numpy.loadtxt(path1)
            ds = f.createCArray(f.root, 'denseFeat', atom, dataToInsert1.shape,filters=filters)
            ### Randomize feats
            numFeat = len(dataToInsert1)
            numBin = math.floor(numFeat/20)
            if numBin > 100:
                dataToInsert=[]
                for i in range(0,20):
                    start =  numBin*(i)
                    if i< 19:
                        end = numBin*(i+1)
                    else:
                        end = numFeat+1
                    temp = dataToInsert1[start:end]
                    numpy.random.shuffle(temp)
                    dataToInsert1[start:end] = temp
                for m in range(0, int(numBin)):
                    for n in range(0,20):
                        ind = int((n*numBin)+m)
                        dataToInsert.append(dataToInsert1[ind,:])
                lastInd =int(20*numBin)
                if numFeat - lastInd > 0:
                    t = int(numFeat)
                    for n1 in range(lastInd,t):
                        dataToInsert.append(dataToInsert1[n1,:])
                ds[:] = dataToInsert
            else:
                numpy.random.shuffle(dataToInsert1)
                ds[:] = dataToInsert1
            ### Close files
            f.flush()
            f.close()


if __name__ == "__main__":

    if (len(sys.argv) != 4):
        print("%s: trajlist indir outdir" % sys.argv[0] )
        exit(1)
    convert(sys.argv[1], sys.argv[2], sys.argv[3])
