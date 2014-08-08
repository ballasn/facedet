import sys
import os


def getInfo(target, info_file):
    """
    Returns the detected bounding boxes associated with the target
    The result is a string
    <info_file> must be an already opened file)
    """
    info = target
    first = False
    found = False
    count = 0
    nb = 0
    for l in info_file.readlines():
        #line = l.split('/')[-1]
        nb += 1
        if first:  # This line contains the number of detected faces
            info += l
            count = int(l[:-1])
            first = False
        elif count != 0:  # Getting location of a face
            info += l
            count -= 1
        elif found:  # We found all info, return the result now
            break
        elif l == target:  # Assuming both contain newline character
            first = True
            found = True

    if not found:
        print repr(target), "was not found"
        info += "0\n"
    return info


def writeOrderedFile(source_file, order_file, output_file):
    """
    Write the info from <source_file> in the order
    given by <order_file>
    <order_file> is assumed to contain one file name per line
    The results is written at <output_file>
    """
    tot = 0
    found = 0
    # Files will be open here
    with open(output_file, "wb") as output:
        with open(order_file, "rb") as order:
            with open(source_file, "rb") as info_file:
                for line in order:
                    tot += 1
                    # Enable to restart reading from the beginning
                    info_file.seek(0)
                    print "looking for", repr(line)
                    info = getInfo(line, info_file)
                    if info[-3:] != '\n0\n':
                        found += 1
                    output.write(info)
                    print "Next file"
    print "Over", tot, "files,", found, "were retrieved."
    print "done"
    return 0


if __name__ == "__main__":
    source_file = "./outputForFDDB.txt"
    order_file = "/data/lisa/data/faces/FDDB/FDDB-folds/FDDB-fold-01.txt"
    output_file = "./orderedOutput.txt"
    #target = '2002/07/26/big/img_517\n'.split('/')[-1]
    print 'source_file :', source_file
    print 'order_file :', order_file
    #print 'target :', repr(target)
    #with open(source_file, "rb") as info_file:
        #    print getInfo(target, info_file)
    writeOrderedFile(source_file, order_file, output_file)
