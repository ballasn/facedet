### Imports
import sys
from os import listdir
from os.path import isfile, join
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

    def extract(self, path_to_image):
        pyramid = {}
        patches = []
        patches_data = []
        ## Load image from path
        image = cv2.imread(path_to_image)
        index = 0
        for s in self.scales:
            pyramid[s] = []
            scaled_size = self.size*s
            scaled_stride = self.stride
            print "scaled_size", scaled_size
            #print "scaled_stride", scaled_stride
            #print index
            for i, x in enumerate(xrange(0, int(image.shape[0]-scaled_size+1),
                    int(scaled_stride))):
                pyramid[s].append([])
                for j, y in enumerate(xrange(0,
                    int(image.shape[1]-scaled_size+1),
                       int(scaled_stride))):
                    # Pick a scaled patch
                    patch = np.copy(image[x:int(x+scaled_size-1),
                        y:int(y+scaled_size-1)])
                    # Resize patch to match self.size
                    patch = cv2.resize(patch, (self.size, self.size), \
                            interpolation = cv2.INTER_CUBIC)
                    # Add the patch to the list
                    patches_data.append(patch)
                    # We need to record the scale to perfom non-max suppression
                    patches.append([path_to_image, [x,y,s]])
                    # pyramid[s][i][j] is now index
                    pyramid[s][i].append(index)
                    index += 1
        return patches_data, patches, pyramid

    def createBatch(self, files):
        """
        Create a batch of patches form a list of files
        The name of corresponding files are written in batch_meta
          with the number of patches for the given file
        """
        batch = []
        batch_meta = []
        for f in files:
            patches_data,patches,pyramid = self.extract(f)
            patches_data = np.array(patches_data,dtype=np.float32)
            batch.extend(patches_data)
            batch_meta.extend(patches)
        return batch, batch_meta, pyramid

### Keep Max of probs non-overlapped
    def nonMaxSuppression(self, pyramid, probs):
        results = copy.copy(probs)
        # near_radius should move given scale
        near_radius = int(self.neighbours + 1)
        local_maxima = []
        # We'll loop over all patches
        # To see if they are local maxima
        for s in self.scales:
            for i in xrange(len(pyramid[s])):
                for j in xrange(len(pyramid[s][i])):
                    center = pyramid[s][i][j]
                    try:
                        c = results[center]
                    except IndexError:
                        continue
                    local_max = True

                # Test the neighbourhood at all scales
                    for s_test in self.scales:
                        # Redefine the closest coordinates at scale s_test
                        i_test, j_test = i*(s/s_test), j*(s/s_test)
                    # Let's scan around (i,j) at the scale s
                    # Defining the neighbourhood
                        near_radius = \
                        int(ceil((self.size*s)/(self.stride*s_test)))
                    # we need to go through [-radius,radius]
                        for di,dj in product(xrange(-near_radius,near_radius+1),
                            xrange(-near_radius, near_radius+1)):
                        # Check if the neighbour exists (border cases)
                            if (i+di)>=0 and (j+dj)>=0 and \
                            (i+di)<len(pyramid[s]) and \
                            (j+dj)<len(pyramid[s][i]):
                            # Compare neighbour and center
                                neighbour = pyramid[s][i+di][j+dj]
                                try:
                                    n = results[neighbour]
                                except IndexError:
                                    print neighbour,i+di,j+dj,s
                                    continue
                                # center is the max
                                if results[center] >= results[neighbour]:
                                    continue
                                else: # This isn't a local max
                                    local_max = False
                                    break
                    if local_max:
                        local_maxima.append(pyramid[s][i][j])
        # Return the list of the indices of patches that are local maxima
        return local_maxima


### Main function

def main(model_file, files, size, scales, stride,classif=False):
    patch_extractor = PatchExtractor(size, scales, stride)
    batch, batch_meta, pyramid = patch_extractor.createBatch(files)
    probs = []

    # Get model
    if classif:
        with open(model_file, "rb") as fd:
                model = cPickle.load(fd)
        print "-"*30
        print model
        print "-"*30
        #for e in batch:
        #    f=batch[10]
        #    print np.array_equal(e,f)
        #    cv2.imshow('image', np.array(e,dtype=np.uint8))
        #    cv2.waitKey(0)

        # Define the classification function
        x = T.tensor4('x')
        classify = function([x], model.fprop(x))
        print "fprop is defined, let's classify"    # Let's classify
        size_minibatch = 128
        splits = (range(i*size_minibatch, min((i+1)*size_minibatch, len(batch)))
                for i in xrange(len(batch)/size_minibatch))
        print len(batch)/size_minibatch + 1,"mini batches"

        for i in xrange(len(batch)/size_minibatch):
            s = range(128*i,min(128*(i+1),len(batch)))
            mini_batch = np.array([batch[i] for i in s])
            mini_batch = np.transpose(mini_batch, (3,1,2,0))
            t = time()
            probs.extend(classify(mini_batch))
            dt = time()-t
            print "Temps par image :", dt/128.0
        # Sur simplet : 0.0411s
        print "Done with classification"
        with open("probs.pkl","wb") as prob_file:
            cPickle.dump(probs,prob_file)
        print "Done dumping"
    else:
        with open("probs.pkl","rb") as prob_file:
            probs = cPickle.load(prob_file)
        print "Done loading"
        # Transform patches for function
        patches_t = []
        for i in xrange(len(probs)):
            x, y, s = batch_meta[i][1]
            patches_t.append([x,y,x+s*patch_extractor.size,y+s*patch_extractor.size])
        probs = [p[0] for p in probs]
        print len(patches_t), len(probs)
        res = patch_extractor.nonMaxSuppression(pyramid, probs)
        ### Transform indices of patch into bounding box
        max_patches = [patches_t[e] for e in res]
        """
        L = {}
        for e,i in zip(max_patches,res):
            L[i]= [j for j in res if patches_t[j]!=e and overlap(e,patches_t[j])]
        selection=[]
        for e in L:
            prob_max = probs[e]
            i_max=e
            for i in L[e]:
                if probs[i]>prob_max:
                    i_max = i
                    prob_max = probs[i]
            selection.append(i_max)
        max_patches = [patches_t[e] for e in selection]
        print max_patches
        """
        displayPatch(batch_meta[0][0], max_patches)

    return 0

### Utils
def overlap(a, b):
    """
    For rectangles defined by 2 points
    """
    return not(a[2]<b[0] or a[3]<b[1] or a[0]>b[2] or a[1]>b[3])

def displayPatch(image_file, patches):
    # Display boudiong box of patches on an image
    image = cv2.imread(image_file)
    for patch in patches:
        cv2.rectangle(image, (patch[0],patch[1]), (patch[2],patch[3]),
                (255,0,0), 3)
    cv2.imshow('image', image)
    c = cv2.waitKey(0)
    while c!="sdfsfsdfd":
        c = cv2.waitKey(0)


def removeIndex(l,i):
    return l[:i]+l[i+1:]

def LocalMaximum():
    #TODO
    raise NotImplementedError


def testSuppression():
    size, scales, stride = 2, [1, 1.5], 1
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
    for i,p in enumerate(probs):
        x, y, s = patches[i][1]
        map_score[s][x][y]=p
    print "map_score[1]"
    print map_score[1]
    print "map_score[1.5]"
    print map_score[1.5]
    # Perform non max suppression
    res = p_ex.nonMaxSuppression(pyramid, probs)
    print "results"
    print res
    for i in res:
        print probs[i]
        print patches[i]


if __name__ == "__main__":
    #testSuppression()

    L = range(10)
    print removeIndex(L,4)
    f = "00001-18107.jpg"
    image = cv2.imread(f)
    print image.shape
    size,scales,stride = 48 ,[4,5], 4
    print "size",size,"stride",stride,"scales",scales
    files = [f]
    modelfile = "../models/facedataset_conv2d_2.pkl"
    main(modelfile, files, size, scales, stride,classif=True)
    main(modelfile, files, size, scales, stride,classif=False)

