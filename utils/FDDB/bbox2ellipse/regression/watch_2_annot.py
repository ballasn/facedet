import cv2
from os.path import isfile
from create_reg_aflw import read_to_dict

# Official annotations
file_annot = 'bbox.txt'
#
file_detect = 'detections.txt'

annot = read_to_dict(file_annot, data='aflw', wh=False)
base_dir = '/data/lisa/data/faces/AFLW/aflw/Images/aflw/data/flickr/'
detect = read_to_dict(file_detect, data='aflw', wh=True)

for e in detect:
    print e
    assert isfile(e)
    img = cv2.imread(e)
    print img.shape
    # BBoxes as [x, y, w, h]
    print 'Official annotations in red, detections in green'
    print 'Annotations :',
    for a in annot[e]:
        print a,
        cv2.rectangle(img, (a[0], a[1]),
                      (a[0]+a[2], a[1]+a[3]),
                      (0, 0, 255), 2)
    if e not in detect:
        print e
        continue
    print '\nDetections :',
    for b in detect[e]:
        print b
        cv2.rectangle(img, (b[1], b[0]),
                      (b[1]+b[3], b[0]+b[2]),
                      (0, 255, 0), 2)

    cv2.imshow(e, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
