"""
This file launches the evaluation on all FDDB.
The file are written so that the script runevaluate.pl
can be executed.
However you may have to check the variables in that exe.
"""

# Imports
import sys
from os.path import join, split, isdir, isfile
from os import remove, mkdir, listdir
from optparse import OptionParser
from cascade import PatchExtractor, getModel

# Args and options
parser = OptionParser()
parser.add_option("--nms", action="store_true", dest="nms", default=False)
(options, args) = parser.parse_args()

if len(args) != 2:
    print "Usage : test_on_FDDB.py <det_dir> <model>"
    sys.exit(2)

else:
    det_dir = args[0]  # Where it will write final results
    model_file = args[1]

# Dir Parameters :
temp_dir = "./temp/"  # Where to store temp files

if not isdir(temp_dir):
    mkdir(temp_dir, 0755)
if not isdir(det_dir):
    mkdir(det_dir, 0755)

# Path_extractor parameters :
size = 48
scales = [1]  # , 1.25, 1.5, 1.75, 2, 2.25, 2.5, 3, 4]
stride = 16
p_e = PatchExtractor(size, scales, stride)
print "-"*30
print "Input size :", size, "|| Stride :", stride, "|| Scales :", scales

if not options.nms:
    print "*"*40
    print "NON MAXIMUM"
    print "            SUPPRESSION"
    print "                        IS NOT ACTIVATED"
    print "*"*40

# Model
print "-"*30
print "Defining the classifier"
model = getModel(model_file)


def processFold(p_e, nb_fold, temp_dir, det_dir, model,
                fold_dir="/data/lisa/data/faces/FDDB/FDDB-folds/",
                img_dir="/data/lisa/data/faces/FDDB/"):
    """
    Apply the model on all images from one fold
    The patches are extracted according to the PatchExtractor properties
    ----------------
    Results written at <det_dir>/fold-<nb_fold>-out.txt
    """
    # define file indicating list of files
    if nb_fold < 10:
        nb_s = "0" + str(nb_fold)
    else:
        nb_s = str(nb_fold)

    fold = join(fold_dir, "FDDB-fold-"+nb_s+".txt")
    print "Working on", fold
    # define list of files as a pyList
    files = []
    with open(fold, "rb") as fold_list:
        for line in fold_list:
            files.append(join(img_dir, line[:-1]+".jpg"))  # Remove \n
    # Checking existing files
    L = []
    for f in files:
        if not isfile(f):
            L.append(f)

    # Faster with NMS
    results = []
    l_f = len(files)
    for i, f in enumerate(files):
        # Extract patches
        sys.stdout.write("\r" + str(nb_fold) + "th fold,"
                         + str(i) + "/" + str(l_f) + " processed images")
        sys.stdout.flush()
        success = p_e.writePatches(f, temp_dir)
        if not success:
            continue

        # Classify
        p_e.classify(model, temp_dir)

        # Format results
        if options.nms:
            results.append([f, p_e.formatResults(f, temp_dir, options.nms)])
        else:
            p_e.formatResults(f, temp_dir, options.nms)

    # Write results for FDDB and perform NMS
    output_fold = join(det_dir, "fold-"+nb_s+"-out.txt")
    p_e.writeResults(files, temp_dir, output_fold, options.nms,
                     results=results)

    # Clean temp_dir from temp files
    print "Now cleaning temp files "
    for f in files:
        # remove the <filename>_clean.pkl files
        filename = split(f)[-1][:-4]
        temp_file = join(temp_dir, filename+"_clean.pkl")
        if isfile(temp_file):
            remove(temp_file)
    print "Done cleaning"
    return L

fold_dir="/data/lisa/data/faces/FDDB/FDDB-folds/"
nb_folds = len(listdir(fold_dir)) / 2
missing = {}
print nb_folds, "to be processed"
for i in range(1, nb_folds + 1):
    missing[i] = processFold(p_e, i, temp_dir, det_dir, model)
print "Missing files :"
print missing

