### Imports
import sys
from os import listdir
from os.path import isfile, join
import cv2
import numpy as np
import copy
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

    def extract(self, path_to_image):
        patches = []
        patches_data = []
        ## Load image from path
        image = cv2.imread(path_to_image)
        for s in scales:
            scaled_size = self.size*s
            scaled_stride = self.stride*s
            for x in xrange(0, image.shape[0], scaled_stride):
                for y in xrange(0, image.shape[1], scaled_stride):
                    # Pick a scaled patch
                    patch = np.copy(image[x:x+scaled_size][y:y+scaled_size])
                    # Resize patch to match self.size
                    patch.resize(patch, (self.size,self.size), interpolation =
                            cv2.INTER_CUBIC)
                    # Add the patch to the list
                    patches_data.append(patch)
                    patches.append(x,y,x+scaled_size,y+scaled_size)
        return patches

### Get probs on patches
##TODO
# get the model from pkl
# loop over images
#   classify patches
## END OF TODO

### Keep Max of probs non-overlapped
def maxFromPatches(patches, probs):
    """
    The idea is to compare patches 2 by 2
    For each pair, if they overlap, we remove the less probable
    That way, a patch only acts on his neighbours.
    """
    max_patches, max_probs = copy.copy(patches), copy.copy(probs)
    for patch, p in zip(max_patches, max_probs):
        for max_patch, max_prob in zip(max_patches, max_probs):
            # Check if we consider the same patch
            if max_patch == patch:
                continue
            # Test if overlapping, in that case compare probs
            # We'll keep the patch if it's more probable
            # and overlapping
            print patch,p,max_patch,max_prob
            print overlap(patch, max_patch), p > max_prob
            if overlap(patch, max_patch) and p > max_prob:
                max_patches.remove(max_patch)
                max_probs.remove(max_prob)
    return max_patches, max_probs

### Main function

def main(files):
    patch_extractor = PatchExtractor(size, scale, stride)
    # Get model
    for f in files:
        patches = patch_extractor.extract(f)
        # probs = Classify patches
        best_patches, best_probs = maxFromPatches(patches, probs)
        # Do something with those results
    return 0

### Utils
def overlap(a, b):
    """
    For rectangles defined by 2 points
    """
    return not(a[2]<b[0] or a[3]<b[1] or a[0]>b[2] or a[1]>b[3])

if __name__=="__main__":
    f = "/u/chassang/sansTFD48.png"
    image = cv2.imread(f)
    c = np.copy(image[:50][:50])
    print "image.shape", image.shape
    print c.shape
    a = [0,0,2,2]
    b = [0,1,1,2]
    print overlap(a,b)
    print overlap(b,a)
    patches =[[0,0,3,3],[0,1,2,2],[0,0,0.9,0.9]]
    probs = [1,1,2]
    print maxFromPatches(patches,probs)
    print patches,probs
    print 0
