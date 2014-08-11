### Imports
import sys
from os import listdir
from os.path import isfile, join, split
import cv2, cv
import numpy as np
import copy
import cPickle
from math import ceil
from theano import function
from theano import tensor as T
from itertools import product
from time import time
from optparse import OptionParser
from emotiw.common.datasets.faces import FDDB



####################################
class PatchExtractor():
    """
    Defines the way patches are extracted from images
    """

    def __init__(self, size, scales, stride):
        self.size = size
        self.scales = scales
        self.stride = stride
        # Radius of the overlapping region at scale 1
        #   given in number of patches
        #   will be used in non-max suppression
        self.neighbours = float(self.size/self.stride)

    def extract(self, path_to_image):
        """
        Extract patches from the image
        """
        pyramid = {}
        patches = []
        patches_data = []
        ## Load image from path
        image = cv2.imread(path_to_image)
        pti = copy.copy(path_to_image)
        filename = split(pti)[-1]
        index = 0
        for s in self.scales:
            scaled_size = self.size*s
            # Image sizes are different
            # A scale may not be used on a given image
            if scaled_size > min(image.shape[:2]):
                #print "scale not used :",s,"on image",path_to_image
                break
            scaled_stride = self.stride*s
            pyramid[s] = []
            #print "scaled_size", scaled_size
            #print "scaled_stride", scaled_stride
            #print index
            i = 0
            y = 0
            ############# Replaced x by y
            while y+scaled_size <= image.shape[1]:
                # pyramid is a dict of dicts of dicts
                pyramid[s].append([])
                j = 0
                x = 0
                while x+scaled_size <= image.shape[0]:
                    # Pick a scaled patch
                    patch = np.copy(image[x:int(x+scaled_size),
                        y:int(y+scaled_size)])
                    # Resize patch to match self.size
                    if patch.shape != (s*self.size, s*self.size,3):
                        print "expected size",(s*self.size, s*self.size)
                        print "returned size",patch.shape
                        sys.exit(1)
                    patch = cv2.resize(patch, (self.size, self.size), \
                            interpolation = cv2.INTER_CUBIC)
                    # Add the patch to the list
                    patches_data.append(patch)
                    # We need to record the scale to perfom non-max suppression
                    patches.append([filename, [int(x),int(y),s]])
                    # pyramid[s][i][j] is now index
                    pyramid[s][i].append(index)
                    index += 1
                    j += 1
                    x += scaled_stride
                i += 1
                y += scaled_stride
                if len(pyramid[s][0])==0:
                    print "scale", s, "patch size", self.size*s
                    print image.shape
                    sys.exit(1)
        return patches_data, patches, pyramid

    def writePatches(self, f, output_dir):
        """
        Create a batch of patches form a file
        The name of corresponding files are written in batch_meta
          with the number of patches for the given file
        """
        filename = split(f)[-1][:-4]
        # Check if real file
        if not isfile(f):
            print f, "is not a file"
            sys.exit(1)
        # Otherwise, process it
        print "Start extracting from "+f
        patches, patches_meta, py = self.extract(f)
        patches = np.array(patches, dtype=np.float32)
        #batch.extend(patches_data)  # [image_data]
        #batch_meta.extend(patches)  # [file,[x,y,s]]
        # One file per pyramid to avoid Memory Error
        py_name = join(output_dir, "py_temp.pkl")
        bm_name = join(output_dir, "batchmeta_temp.pkl")
        b_name  = join(output_dir, "batch_temp.npy")
        print "got", patches.shape[0], "patches"
        print "wrrting files :",py_name, bm_name, b_name

        with open(py_name, "wb") as py_file:
            cPickle.dump(py, py_file)
        with open(bm_name, "wb") as bm_file:
            cPickle.dump(patches_meta, bm_file)  #[file, [x,y,s]]
        np.save(b_name, patches)
        return 0

### Keep Max of probs non-overlapped
    def nonMaxSuppression(self, pyramid, probs):
        """
        We'll loop over all patches
        To see if they are local maxima
        Loop over the index of pyramids
        """
        local_maxima = {}
        results = copy.copy(probs)

        for s in pyramid:
            for i in xrange(len(pyramid[s])):
                for j in xrange(len(pyramid[s][i])):
                    #print "sij",s,i,j,
                    s_d = s
                    i_d = i
                    j_d = j
                    center = pyramid[s][i][j]
                    try:
                        c = results[center]
                        if c == 0.0:
                            #print "is zero"
                            continue
                        #else:
                            #print "______ New Center ____"
                            #print "center",s,i,j,":",center
                    except KeyError:
                        #print "center",s,i,j,":",center
                        continue
                    local_max = True

                    # Define the center patch
                    p_x = i*s*self.stride
                    p_y = j*s*self.stride
                    p = [p_x, p_y, p_x + s*self.size, p_y + s*self.size]
                    #print "center bbox",p
                    #print "center score",c
                    nei = []

                    # Test the neighbourhood at all scales
                    for s_test in pyramid:
                        if not local_max:
                            break
                        #print s_test
                        i_t = 0
                        j_t = 0
                        size_t = s_test*self.size
                        stride_t = s_test*self.stride
                        p_test = [0, 0, size_t, size_t]
                        indices = []
                        maxi, maxj = 0, 0
                        # Scan new scale until overlap
                        while not overlap(p, p_test):
                            if i_t == len(pyramid[s_test]):
                                break
                            elif j_t == len(pyramid[s_test][i_t]):
                                j_t = 0
                                i_t += 1
                                maxi = max(maxi, i_t)
                            else:
                                j_t += 1
                                maxj = max(maxj, j_t)
                            p_x_t = i_t*stride_t
                            p_y_t = j_t*stride_t
                            p_test = [p_x_t, p_y_t, p_x_t + size_t, p_y_t + size_t]
                            indices.append(p_test)
                            # Moving on x-axis
                        #if s==s_test:
                            #print "p_test",
                            #print p_test

                        if i_t==len(pyramid[s_test]) or \
                                j_t==len(pyramid[s_test][i_t]):  # no overlap
                            print "_"*20
                            print "no overlap found"
                            print p
                            print "s,  i,  j"
                            print s, i, j
                            print "s_test,i_t,j_t"
                            print s_test, i_t, j_t
                            print maxi,maxj
                            print "len py[s_test],py[s_test][i_t]"
                            print \
                            len(pyramid[s_test]),len(pyramid[s_test][i_t-1])
                            print "len py[s],py[s][i]"
                            print \
                            len(pyramid[s]),len(pyramid[s][i])
                            # print indices
                            print "Stopping program at line 194"
                            sys.exit(1)

                        p_test_var = copy.copy(p_test)
                        di = 0
                        dj = 0

                            # Loop over x-axis
                        while overlap(p, p_test_var) and \
                                i_t + di < len(pyramid[s_test]):
                            if not local_max:
                                break

                            # Loop over y-axis
                            while overlap(p, p_test_var) and \
                                    j_t + dj < len(pyramid[s_test][i_t]):
                                if not local_max:
                                    break

                                # Compare neighbour and center
                                if (s, i, j) == (s_test, i_t+di, j_t+dj):
                                    #print (s,i,j), (s_test, i_t+di, j_t+dj)
                                    p_test_var = [p_test[0] + di*stride_t,
                                                  p_test[1] + dj*stride_t,
                                                  p_test[2] + di*stride_t,
                                                  p_test[3] + dj*stride_t]
                                    #print "found myself",p_test_var
                                    dj += 1

                                else:
                                    neighbour = pyramid[s_test][i_t+di][j_t+dj]

                                    try:
                                        n = results[neighbour]
                                        #print "neighbour :",s_test,i_t+di,j_t+dj,
                                    except KeyError:
                                        dj += 1
                                        p_test_var = [p_test[0] + di*stride_t,
                                                      p_test[1] + dj*stride_t,
                                                      p_test[2] + di*stride_t,
                                                      p_test[3] + dj*stride_t]
                                        continue

                                    if c >= n:
                                        results[neighbour] = 0.0
                                    else: # This isn't a local max
                                        #print "isn't a local max"
                                        #print "-----------------"
                                        results[center] = 0.0
                                        local_max = False
                                    # Next patch on y-axis
                                    dj += 1
                                    p_test_var = [p_test[0] + di*stride_t,
                                                  p_test[1] + dj*stride_t,
                                                  p_test[2] + di*stride_t,
                                                  p_test[3] + dj*stride_t]
                                # Next patch on x-axis
                            dj = 0
                            di += 1
                            p_test_var = [p_test[0] + di*stride_t,
                                          p_test[1] + dj*stride_t,
                                          p_test[2] + di*stride_t,
                                          p_test[3] + dj*stride_t]

                        # center passed tests in its neighbourhood
                        # It's a local max
                        # Store it as [x,y,w,h,p]
                    #print "sij fin",s,i,j
                    if (s,i,j)!=(s_d,i_d,j_d):
                        print (s,i,j),"should be",(s_d,i_d,j_d)
                        sys.exit(1)
                    if local_max:
                        #print center, [p_x, p_y, int(s*self.size),
                                #int(s*self.size)]
                        #print p
                        local_maxima[center] = [p_x, p_y, int(s*self.size),
                            int(s*self.size), c]
                        #print "---------------------- > is a Local Max"
        print local_maxima
        # Return the list of [x,y,w,h,p] of patches that are local maxima
        return local_maxima

    def formatResults(self, data_dir, f):
        """
        This function formats the result, so that
        they can be read to create the final life.
        If NMS is activated it will be perform it
        before writing the file.
        """
        # Get the filename
        filename = split(f)[-1][:-4]
        pyramid_file = join(data_dir, "py_temp.pkl")
        classif_file = join(data_dir, filename+"_classif.pkl")

        # Loading the pyramid dict
        if not isfile(pyramid_file):
            print pyramid_file, "is not a valid file"
            sys.exit(1)
        if not isfile(classif_file):
            print classif_file, "is not a valid file"
            sys.exit(1)

        with open(classif_file, "rb") as c_file:
            classifications = cPickle.load(c_file)
        print "classif", len(classifications)
        indices = []

        ######### NMS ###########
        if options.nms:
            with open(pyramid_file, "rb") as p_file:
                pyramid = cPickle.load(p_file)
            probs = {}
            for i in classifications:
                probs[i] = classifications[i][4]
            # Apply NMS
            print "Applying nonMaxSuppression"
            cleaned_results = self.nonMaxSuppression(pyramid, \
                            probs)
        else:
            cleaned_results = classifications
        with open(join(data_dir, filename + "_clean.pkl"), "wb") as results_file:
            cPickle.dump(cleaned_results, results_file)
        return 0




    def writeResults(self, files, data_dir, output_file):
        """
        Write classification to a file.
        This file aims at being read for FDDB test.
        <image_name>
        <number of faces in image>
        <left_x top_y width height detection_score>
        ....
        <left_x top_y width height detection_score>
        -------------------------------------------
        classifications = list /
        classifications[0] = [f,x,y,w,h,p(face)]
        """
        # Checking sizes
        c = 0
        indices = {}
        patches_per_file = {}
        ## Now tranforming results file by file
        for f in files:
            if not isfile(join(data_dir,f)):
                print "ignoring invalid file :", f
                continue
            filename = split(f)[-1][:-4]
            clean_file = join(data_dir, filename+"_clean.pkl")
            with open(clean_file, "rb") as c_file:
                cleaned_results = cPickle.load(c_file)
            for e in cleaned_results:
                # Need to tranform bouding box to fit display
                c = [ int(i) for i in cleaned_results[e][:4] ]
                c.append(cleaned_results[e][4])
                if f in patches_per_file:
                    patches_per_file[f].append(c)
                else:
                    patches_per_file[f] = [c]
                    #[x,y,w,h,p]
            print "got", len(cleaned_results), "for", filename

        # Write bounding box and prob in the file
        with open(output_file, "wb") as output:
            print "writing", output_file
            for e in patches_per_file:
                # We need to format names to fit FDDB test
                # We remove /data/lisa/data/faces/FDDB and the extension
                n = e.split("/")[6:]
                n = "/".join(n)[:-4]
                output.write(n+"\n") # Filename for FDDB
                output.write(str(len(patches_per_file[e]))+"\n") # Nb of faces
                # Faces for the image
                for p in patches_per_file[e]:
                    line = ' '.join([str(e) for e in p])
                    output.write(line+"\n")
        return 0


    def classify(self, fprop_func, data_dir, f):

        size_minibatch = 128
        t0 = time()
        probs = {}
        info = {}

        # Load data from patches
        filename = split(f)[-1]
        ba_file = join(data_dir, "batch_temp.npy")
        # Always check the file
        if not isfile(ba_file):
            print "can't read", ba_file
            sys.exit(1)
        # Load data
        batch = np.load(ba_file)
        # Load metadata
        with open(join(data_dir, "batchmeta_temp.pkl"), "rb") as bm_file:
            batch_meta = cPickle.load(bm_file)
        nb_minibatch = int(ceil(float(len(batch))/float(size_minibatch)))
        print "nb_minibatch",nb_minibatch,len(batch)/size_minibatch+1

        for i in xrange(nb_minibatch):
            s = range(128*i,min(128*(i+1),len(batch)))
            mini_batch = np.array([batch[i] for i in s])
            mini_batch = np.transpose(mini_batch, (3,1,2,0))
            t = time()
            # Classify returns a couple [p(face), p(non-face)]
            # We'll keep those with p(face)>0.5
            cl = fprop_func(mini_batch)
            for j in s:
                p =  cl[j%128][0]
                if p > 0.5:
                    # Add element to the dict of possible faces
                    f, [x, y, s] = batch_meta[j]
                    # Debug verification
                    if type(x) == float or type(y) == float:
                        print "x",x,"y",y
                        print "one is a float, they should be ints"
                        sys.exit(1)
                    info[j] = ([x, y, int(s*self.size),\
            int(s*self.size), p])  # [f,x,y,w,h,prob]
            dt = time()-t
            print "Temps par patch :", dt/128.0
            # Sur simplet : 0.0411s
            # Sur GTX480 : 0.0006s
        # Writing results
        with open(join(data_dir, filename[:-4]+"_classif.pkl"), "wb") as result_file:
            cPickle.dump(info, result_file)
        print "Wrote results in", join(data_dir,filename[:-4]+"_classif.pkl")
        print "Wrote a list of", len(info), "elements"
        return 0

def getModel(model_file):
    """
    Loads the model and returns it
    """
    with open(model_file, "rb") as fd:
            model = cPickle.load(fd)
    print "-"*30
    print model
    print "-"*30
    # Define the classification function
    x = T.tensor4('x')
    classify = function([x], model.fprop(x))
    print "fprop is defined, ready to use"
    return classify

def displayResults(prefix, results_file):
    """
    Get results from txt file
    Display bounding boxes on images
    """
    #with open(pyramid_file,"rb") as p_file:
    #    pyramid = cPickle.load(p_file)
    #print "Done loading pyramid"

    best_patches = {}
    with open(results_file, "rb") as r_file:
        count = 0
        for line in r_file:
            line = line[:-1]  # Removing \n in the end
            if isfile(join(prefix,line)+".jpg"):
                if not count == 0:
                    print "count", count, "should be 0"
                f = join(prefix,line)+".jpg"
                best_patches[f] = []
            else:
                line = line.split()
                if len(line) == 1:
                    count = int(line[0])
                else:
                    line = [int(l) for l in line[:-1]]
                    best_patches[f].append([line[0], line[1], line[0]+line[2],
                        line[1]+line[3]])
                    count -= 1

    for f in best_patches:
        displayPatch(f, best_patches[f])
    return 0


def displayPatch(image_file, patches):
    # Display bounding box of patches on an image
    if not isfile(image_file):
        print image_file,"is not a valid file"
        sys.exit(2)
    image = cv2.imread(image_file)
    for patch in patches:
        # Here we used to transpose them
        #cv2.rectangle(image, (patch[1],patch[0]), (patch[3],patch[2]),
                #(0,255,0), 3)
        cv2.rectangle(image, (patch[0],patch[1]), (patch[2],patch[3]),
                (0,255,0), 3)
    cv2.imshow(image_file, image)
    c = cv2.waitKey(0)
    cv2.destroyWindow(image_file)
    return 0

def overlap(a, b):
    """
    For rectangles defined by 2 points
    """
    return not(a[2]<b[0] or a[3]<b[1] or a[0]>b[2] or a[1]>b[3])

################################################
########### TEST FUNCTIONS #####################
################################################

def testSuppression():
    size, scales, stride = 2, [1.0, 1.5, 2], 1
    print "size", size, "stride", stride, "scales", scales
    p_ex = PatchExtractor(size, scales, stride)
    data = np.array([[[1,1,1],[1,1,1],[1,1,1],[2,2,2]],
                    [[2,2,2],[1,1,1],[1,1,1],[1,1,1]],
                    [[1,1,1],[1,1,1],[4,4,4],[4,4,4]],
                    [[2,2,2],[1,1,1],[4,4,4],[4,4,4]]],dtype=np.int8)
    img = cv.fromarray(data)
    cv.SaveImage("a.png",img)
    # was written at a.png
    img_path = "a.png"
    # Get patches
    patches_data, patches, pyramid = p_ex.extract(img_path)
    probs = []
    print "patches coords"
    print len(patches)
    # Create some score
    for p in patches_data:
        probs.append(np.amax(p))
    print "probs"
    print probs
    # Map scores to see them
    map_score = {}
    map_score[1] = np.zeros((3,3))
    map_score[1.5] = np.zeros((2,2))
    map_score[2.0] = np.zeros((1,1))

    for i,p in enumerate(probs):
        x, y, s = patches[i][1]
        map_score[s][x][y]=p
    print "pyramid"
    print pyramid
    print "\nmap_score[1]"
    print map_score[1]
    print "\nmap_score[1.5]"
    print map_score[1.5]
    print "\nmap_score[2]"
    print map_score[2]

    # Perform non max suppression
    res = p_ex.nonMaxSuppression(pyramid, probs)
    print "\nresults of non Max Suppression"
    for i in res:
        print probs[i], patches[i]

def testOverlap(n,p):
    I = range(n)
    J = range(p)
    B = [5,5,6,6]
    for i in range(n):
        for j in range(p):
            A = [i,j,i+3,j+3]
            if overlap(A,B):
                print A
    return 0

def testRealCases():
    size,scales,stride = 48 ,[3,4,5,6,7], 8
    print "size",size,"stride",stride,"scales",scales
    patch_extractor = PatchExtractor(size, scales, stride)


# Testing with some files from FDDB
    folder = "/data/lisa/data/faces/FDDB_old/2002/07/19/big/"
    files = [join(folder, f) for f in
            listdir(folder) if
            isfile(join(folder, f))]
    print files

    ### Define used files
    modelfile = "../models/facedataset_conv2d_2.pkl"
    classif_file = "classif.pkl"  # Can grow large
    pyramid_file = "pyramids.pkl"  # Can grow large
    output_file = "outputForFDDB.txt"

    ### Classify
    p2 = classify(modelfile, files, patch_extractor)
    ### Write the results for FDDB test
    print "-"*30
    print "Writing results in file"
    print "-"*30
    patch_extractor.writeResults(classif_file, pyramid_file, output_file)
    #### Display results
    print "-"*30
    print "Display results"
    print "-"*30
    displayResults(prefix, output_file)

######### Arguments ##############

#Non Images
parser = OptionParser()
parser.add_option("--nms", action="store_true", dest="nms",default=False)
(options, args) = parser.parse_args()
print "Non-max Suppression :", options

## Define the patch_extractor
size = 48
scales = [1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 3, 4]
#scales = [1,2]
stride = 24
print "Input size :", size
print "Stride :", stride
print "Scales :", scales
p_e = PatchExtractor(size, scales, stride)

"""### Look for the images
list_file = args[0]
prefix = args[1]
files = []
with open(list_file,"rb") as l_file:
    for line in l_file:
        files.append(join(prefix, line[:-1])+".jpg")
print "got", len(files), "files"
#files = [files[0]]
print files[:2]
"""
# Using FDDB interface by Thomas Rohee
fddb = FDDB.FDDB()
print "Nombre de fichiers :", len(fddb)
model_file = "../models/facedataset_700k_0408.pkl"
classif_file = "classif.pkl"  # Can grow large
data_dir = "/data/lisatmp3/chassang/facedet/patches"  # Can grow large
output_file = "outputForFDDB.txt"

########## Pipeline is now used one file at a time
# 1. Get the model
print "-"*30
print "Defining the classifier"
model = getModel(model_file)

# 2. Loop over files to extract and classify
#    Temp files are rewritten to limit mem usage

for i in xrange(10):#len(fddb)):
    # Get image path
    f = fddb.get_original_image_path_relative_to_base_directory(i)
    f = join(fddb.absolute_base_directory, f)
    print f
    # 2.1 Extract patches and write them at <data_dir>
    print "-"*20
    print "Creating batches"
    print "-"*20
    p_e.writePatches(f, data_dir)

    # 2.2 Classify
    print "-"*20
    print "Classifying"
    print "-"*20
    p_e.classify(model, data_dir, f)

    # 2.3 Format results
    print "-"*20
    print "Formatting"
    print "-"*20
    p_e.formatResults(data_dir, f)

# 3. Write results for FDDB
#    Perform NMS if specified
print "-"*20
print "Writing Results and performing NMS"
print "-"*20
p_e.writeResults(files[:10], data_dir, output_file)

# 4. Display results
print "-"*20
print "Displaying"
print "-"*20
displayResults(prefix, output_file)


