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
### Arguments

if len(sys.argv)<2:
    print "cascade.py <folder>"
    #sys.exit(2)
else:
    folder = sys.argv[1]
    images = [f for f in listdir(folder) if isfile(join(folder,f)) ]
    #Non Images files ?


# Get patches from image
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

    def extract(self, path_to_image,start=0):
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
        index = start
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
                    patches.append([filename, [x,y,s]])
                    # pyramid[s][i][j] is now index
                    pyramid[s][i].append(index)
                    index += 1
                    j += 1
                    x += scaled_stride
                i += 1
                y += scaled_stride
                if len(pyramid[s][0])==0:
                    print "scale",s,"patch size",self.size*s
                    print image.shape
                    sys.exit(1)
        return patches_data, patches, pyramid, index

    def createBatch(self, files):
        """
        Create a batch of patches form a list of files
        The name of corresponding files are written in batch_meta
          with the number of patches for the given file
        """

        batch = []
        batch_meta = []
        pyramids = {}
        start = 0
        for f in files:
            print "start",start
            patches_data, patches, py, start = self.extract(f,start)
            patches_data = np.array(patches_data, dtype=np.float32)
            batch.extend(patches_data)  # [image_data]
            batch_meta.extend(patches)  # [file,[x,y,s]]
            filename = split(f)[-1]
            pyramids[filename] = py
        return batch, batch_meta, pyramids

### Keep Max of probs non-overlapped
    def nonMaxSuppression(self, pyramid, probs):
        """
        We'll loop over all patches
        To see if they are local maxima
        Loop over the index of pyramids
        """
        local_maxima = []
        results = copy.copy(probs)

        for s in pyramid:
            for i in xrange(len(pyramid[s])):
                for j in xrange(len(pyramid[s][i])):
                    center = pyramid[s][i][j]
                    try:
                        c = results[center]
                        if c == 0.0:
                            #print "is zero"
                            continue
                        #else:
                            # print "______ New Center ____"
                           # print "center",s,i,j,":",center
                    except IndexError:
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
                        # Scan new scale until overlap
                        while not overlap(p, p_test):
                            if i_t == len(pyramid[s_test]):
                                break
                            elif j_t == len(pyramid[s_test]):
                                j_t = 0
                                i_t += 1
                            else:
                                j_t += 1
                            p_x = i_t*stride_t
                            p_y = j_t*stride_t
                            p_test = [p_x, p_y, p_x + size_t, p_y + size_t]
                            indices.append(p_test)
                            # Moving on x-axis
                        #if s==s_test:
                            #print "p_test",
                            #print p_test
                        if i_t==len(pyramid[s_test]) or \
                                j_t==len(pyramid[s_test][i_t]):
                                print "_"*20
                                print "no overlap found"
                                print p
                                print "s_test,i_t,j_t", s_test, i_t, j_t
                               # print indices
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
                                    dj += 1
                                    p_test_var = [p_test[0] + di*stride_t,
                                                  p_test[1] + dj*stride_t,
                                                  p_test[2] + di*stride_t + size_t,
                                                  p_test[3] + dj*stride_t + size_t]
                                    #print "found myself",p_test_var

                                else:
                                    neighbour = pyramid[s_test][i_t+di][j_t+dj]

                                    try:
                                        n = results[neighbour]
                                        #######################
                                        nei.append((s_test,i_t+di,j_t+dj))
                                        #######################

                                        #print "neighbour :",s_test,i_t+di,j_t+dj,
                                    except IndexError:
                                        print "neighbour", s_test, i_t+di, j_t+dj, ":", neighbour
                                        print (s_test, i_t+di, j_t+dj)
                                        sys.exit(1)

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
                                                  p_test[2] + di*stride_t + size_t,
                                                  p_test[3] + dj*stride_t + size_t]
                            # Next patch on x-axis
                            dj = 0
                            di += 1
                            p_test_var = [p_test[0] + di*stride_t,
                                          p_test[1] + dj*stride_t,
                                          p_test[2] + di*stride_t + size_t,
                                          p_test[3] + dj*stride_t + size_t]

                    # center passed tests in its neighbourhood
                    # It's a local max
                    if local_max:
                        local_maxima.append(center)
                        #print nei
                        #print "---------------------- > is a Local Max"
        # Return the list of the indices of patches that are local maxima
        return local_maxima

    def writeResults(self, classifications_file, pyramid_file, output_file):
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
#### As it uses dict structure, it won't write twice for the same file
        with open(classifications_file, "rb") as c_file:
            classifications = cPickle.load(c_file)
        with open(pyramid_file, "rb") as p_file:
            pyramids = cPickle.load(p_file)

        # Checking sizes
        c = 0
        for f in pyramids:
            for s in pyramids[f]:
                for p in pyramids[f][s]:
                    c += len(p)
        print "size of pyramid", c
        print "len of classif", len(classifications)

    # pyramids is first indexed by the file name
        cleaned_results = {}
        for f in pyramids:
            print f
            cleaned_results[f] = self.nonMaxSuppression(pyramids[f], \
                    [c[5] for c in classifications])
        print cleaned_results
        patches_per_file = {}
        ## Now tranforming results file by file
        for f in cleaned_results:
            print f
            for i in cleaned_results[f]:
                e = classifications[i]
                print e
                # Need to tranform bouding box to fit display
                if f in patches_per_file:
                    patches_per_file[f].append(e[1:])
                else:
                    patches_per_file[f] = [e[1:]]
                    #[x,y,w,h,p]

        # Write bounding box and prob in the file
        with open(output_file,"wb") as output:
            for e in patches_per_file:
                output.write(e+"\n") # Filename
                output.write(str(len(patches_per_file[e]))+"\n") # Nb of faces
                # Faces for the image
                for p in patches_per_file[e]:
                    line = ' '.join([str(e) for e in p])
                    output.write(line+"\n")
        return 0


### Main function

def classify(model_file, files, patch_extractor, output_file):
    batch, batch_meta, pyramids = patch_extractor.createBatch(files)
    probs = []
    ####### Get model
    with open(model_file, "rb") as fd:
            model = cPickle.load(fd)
    print "-"*30
    print model
    print "-"*30
    # Define the classification function
    x = T.tensor4('x')
    classify = function([x], model.fprop(x))
    print "fprop is defined, let's classify"
    ### For one file
    size_minibatch = 128
    print len(batch)/size_minibatch + 1, "mini batches"
    t0 = time()
    for i in xrange(len(batch)/size_minibatch+1):
        s = range(128*i,min(128*(i+1),len(batch)))
        mini_batch = np.array([batch[i] for i in s])
        mini_batch = np.transpose(mini_batch, (3,1,2,0))
        t = time()
        # Classify returns a couple [p(face), p(non-face)]
        cl = classify(mini_batch)
        probs.extend([p[0] for p in cl])
        dt = time()-t
        print "Temps par image :", dt/128.0

    # Sur simplet : 0.0411s
    # Sur GTX480 : 0.0006s
    print "File :"
    print "Classified", len(batch), "patches in", time()-t0

    ####### Write the results with the info about patches and files
    # Transform patches for function
    info = []
    for i in xrange(len(probs)):
        f, [x, y, s] = batch_meta[i]
        info.append([f, x, y, s*patch_extractor.size,\
            s*patch_extractor.size, probs[i]])  #[f,x,y,w,h,prob]

        # Result file is large
    with open(output_file, "wb") as result_file:
        cPickle.dump(info, result_file)

    with open("pyramids.pkl", "wb") as pyramid_file:
        cPickle.dump(pyramids, pyramid_file)

    print "Done dumping"
    return pyramids


def displayResults(folder, results_file):
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
            if isfile(join(folder,line)):
                if not count == 0:
                    print "count", count, "should be 0"
                f = join(folder,line)
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

###### Utils
def overlap(a, b):
    """
    For rectangles defined by 2 points
    """
    return not(a[2]<b[0] or a[3]<b[1] or a[0]>b[2] or a[1]>b[3])

def displayPatch(image_file, patches):
    # Display bounding box of patches on an image
    if not isfile(image_file):
        print image_file,"is not a valid file"
        sys.exit(2)
    image = cv2.imread(image_file)
    for patch in patches:
        # Here we transpose them
        cv2.rectangle(image, (patch[1],patch[0]), (patch[3],patch[2]),
                (0,255,0), 3)
        cv2.rectangle(image, (patch[0],patch[1]), (patch[2],patch[3]),
                (255,0,0), 1)
    cv2.imshow(image_file, image)
    c = cv2.waitKey(0)
    cv2.destroyWindow(image_file)
    return 0

def removeIndex(l,i):
    return l[:i]+l[i+1:]

    return 0

################################## TEST
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

if __name__ == "__main__":
    #testSuppression()
    #testOverlap(10,10)

    f1 = "00001-18107.jpg"
    f2 = "image.jpg"
    f3 = "image2.jpg"
    f4 = "image3.jpg"
    f5 = "image4.jpg"
    image = cv2.imread(f1)
    print image.shape
    size,scales,stride = 48 ,[3,4,5,6,7], 8
    print "size",size,"stride",stride,"scales",scales
    patch_extractor = PatchExtractor(size, scales, stride)
    # Testing with some files from FDDB
    folder = "/data/lisa/data/faces/FDDB_old/2002/07/19/big/"
    #files = ["/data/lisa/data/faces/FDDB_old/2002/07/19/big/img_827.jpg"]
    #files = [f1,f2,f3,f4,f5]

    files = [join(folder,f) for f in
            listdir(folder) if
            isfile(join(folder,f))]
    print files

    ### Define used files
    modelfile = "../models/facedataset_conv2d_2.pkl"
    classif_file = "classif.pkl" # Can grow large
    pyramid_file = "pyramids.pkl" # Can grow large
    output_file = "outputForFDDB.txt"
    ### Classify

    p2 = classify(modelfile, files, patch_extractor, classif_file)
    ### Write the results for FDDB test
    print "-"*30
    print "Writing results in file"
    print "-"*30
    patch_extractor.writeResults(classif_file, pyramid_file, output_file)
    #### Display results
    print "-"*30
    print "Display results"
    print "-"*30
    displayResults(folder, output_file)

