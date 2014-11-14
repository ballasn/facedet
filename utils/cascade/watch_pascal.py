import cv2
from os.path import join
import numpy as np

# Pascal
base_dir = "/data/lisa/data/faces/AFW/testimages/"
#file_ = "../AFW/face-eval/detections/AFW/Ours_Headhunter.txt"
file_ = '../AFLW/AFLW_files/new_annotations.txt'
pb = False
with open(file_, 'r') as f:
    cur = 0
    lines = f.read().splitlines()
    print 'Got ', len(lines), 'lines'

    # Initialize the first image
    row = list(lines[0].split(' '))
    cur_img = str(row[0])
    print row
    if not cur_img[-4:] == '.jpg':
        cur_img += '.jpg'
    image_file = join(base_dir, cur_img)
    print image_file
    img_ = cv2.imread(image_file)
    sh = img_.shape
    rs = min(400.0/img_.shape[0], 400.0/img_.shape[1])
    print rs
    img_ = cv2.resize(img_, (int(rs * img_.shape[1]), int(rs*img_.shape[0])),
                      interpolation=cv2.INTER_CUBIC)

    for i in range(len(lines)):

        row = list(lines[i].split(' '))
        # Get the path to the file
        new_img = str(row[0])

        # If different than the previous one
        # Display everything and set up the new img
        if new_img != cur_img:
            print 'last :', row
            cv2.imshow(image_file[len(base_dir):], img_)
            if pb:
                print pb_coords
                print pb_check
                print img_.shape
                print pb_row[2:]
                print sh
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            pb = False

            # Define the new reference
            cur_img = str(row[0])
            if not cur_img[-4:] == '.jpg':
                cur_img += '.jpg'
            image_file = join(base_dir, cur_img)
            img_ = cv2.imread(image_file)
            sh = img_.shape
            rs = min(600.0/img_.shape[0], 600.0/img_.shape[1])
            print '*'*80
            print row
            print image_file
            print img_.shape
            print rs
            img_ = cv2.resize(img_, (int(rs * img_.shape[1]), int(rs*img_.shape[0])),
                              interpolation=cv2.INTER_CUBIC)

        # Process the bbox
        print "img :", new_img, "cur_img :", cur_img
        if len(row) == 6:
            coords = [int(rs * float(r)) for r in row[2:]]
        elif len(row) == 5:
            coords = [int(rs * float(r)) for r in row[1:]]
        print coords
        check = [coords[i] > sh[i%2] for i in range(len(coords))]
        score = float(row[1])
        cv2.rectangle(img_, (coords[0], coords[1]),
                      (coords[2], coords[3]),
                      (0, int(score * 15), 255 - int(score *15)), 2)
        if any(check):
            pb_coords = list(coords)
            pb_check = list(check)
            pb_row = list(row)
            pb = True



