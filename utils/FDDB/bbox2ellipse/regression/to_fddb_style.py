import sys


def bboxToDict(file_name, base_dir=''):
    '''
    Returns the dict of bboxes indexed by their file_id
    '''
    with open(file_name, 'r') as f:
        lines = f.read().splitlines()
    rval = {}
    print 'Got', len(lines), 'lines'
    for line in lines:
        # Assume line = 'id x y w h'
        row = line.split(' ')
        if base_dir+row[0] not in rval:
            rval[base_dir+row[0]] = []
        coords = [int(float(e)) for e in row[1:]]
        rval[base_dir+row[0]].append(coords)
    s = 0
    si = 0
    for e in rval:
        s += len(rval[e])
        si += 1
    print s, si
    assert s == len(lines)
    return rval


def dictToFddbStyle(boxes,
        base_dir='/data/lisa/data/faces/AFLW/aflw/Images/aflw/data/flickr/'):
    s = ''
    for e in boxes:
        if not base_dir in e:
            f = base_dir+e
        else:
            f = e
        s += str(f) + '\n'
        s += str(len(boxes[e])) + '\n'
        for box in boxes[e]:
            s += ' '.join(map(str, box))+'\n'
    return s


if __name__ == '__main__':

    if len(sys.argv) < 3:
        print "usage %s : <input_file> <output_file>"\
              % sys.argv[0]
        sys.exit(2)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    base_dir='/data/lisa/data/faces/AFLW/aflw/Images/aflw/data/flickr/'
    #base_dir=''
    print 'Gathering data from', input_file, 'in a dict...'
    print 'adding', base_dir, 'to keys'
    dict_box = bboxToDict(input_file, base_dir=base_dir)
    print ' done !'
    print 'Writing at', output_file, '...',
    s = dictToFddbStyle(dict_box, base_dir=base_dir)
    with open(output_file, 'w') as out:
        out.write(s)
    print ' done !'
