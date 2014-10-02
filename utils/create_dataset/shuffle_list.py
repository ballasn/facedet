import sys
from random import shuffle
if len(sys.argv) != 2:
    print 'usage : %s <text_file>' % sys.argv[0]
    sys.exit(2)

text = sys.argv[1]
with open(text, 'r') as t:
    l = t.readlines()

shuffle(l)
with open(text[:-4]+'_shuffled.txt', 'w') as t_s:
    for e in l:
        t_s.write(e)

print 'Done'

