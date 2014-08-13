import sys
from os.path import isdir, join
from os import mkdir, listdir

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
    source_dir = "./detections_obtained"
    order_dir = "/data/lisa/data/faces/FDDB/FDDB-folds/"
    output_dir = "./detections_ordered"
    if not isdir(output_dir):
        mkdir(output_dir)
    print 'source_dir :', source_dir
    print 'order_dir :', order_dir
    print 'output_dir :', output_dir
    nb_folds = len(listdir(order_dir)) / 2
    for i in range(1, nb_folds+1):
        if i < 10:
            nb_s = "0" + str(i)
        else:
            nb_s = str(i)
        source_file = join(source_dir, "fold-"+nb_s+"-out.txt")
        order_file = join(order_dir, "FDDB-fold-"+nb_s+".txt")
        output_file = join(output_dir, "fold-"+nb_s+"-out.txt")
        writeOrderedFile(source_file, order_file, output_file)
