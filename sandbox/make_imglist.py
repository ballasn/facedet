from os import listdir
from os.path import isfile, join


def filelist(mypath):
    onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
    return onlyfiles

if __name__=='__main__':
    mypath = '/data/lisa/data/faces/AFLW/aflw/Images/aflw/data/flickr/'
    l = filelist(mypath)
    with open('image_list.txt','w') as outfile:
        for fil in l:
            outfile.write(str(fil)+'\n')
    print 'File has been written'
            
