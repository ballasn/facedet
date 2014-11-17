import sys
import os
import cv2
import csv


if __name__ == '__main__':

    if len(sys.argv) < 4:
        print("Usage %s: <ubi_csv> <list> <annot>" % sys.argv[0])
        print("Takes a csv files and output FDDB style annotation files")
        sys.exit(1)

    basedir = "/data/lisatmp3/ballasn/Ubiface/"

    finput = sys.argv[1]
    flist = sys.argv[2]
    fannot = sys.argv[3]

    fd_list = open(flist, 'w')
    fd_annot = open(fannot, 'w')

    with open(finput, 'r') as fin:
        annot = csv.reader(fin, delimiter=',')
        for elem in annot:
            print elem
            img_path, x, y, lx, ly = elem
            img_path = os.path.join(basedir, img_path)
            score = 1.0


            img = cv2.imread(img_path)

            #x = img.shape[1] - float(x)
            #y = img.shape[0] - float(y)
            x = float(x)
            y = float(y)
            lx = float(lx)
            ly = float(ly)

            cv2.rectangle(img,
                          (int(x), int(y)), (int(x+lx), int(y+ly)),
                          (0, 0, 255), 2)
            #cv2.imshow(img_path, img)
            #cv2.waitKey(2)
            #cv2.destroyAllWindows()

            fd_list.write(img_path + '\n')

            fd_annot.write(img_path + '\n')
            fd_annot.write("1\n")
            fd_annot.write(str(x) + " " + str(y) + " " + str(lx) + " " + str(ly) + " 1.0\n")

    fd_list.close()
    fd_annot.close()









