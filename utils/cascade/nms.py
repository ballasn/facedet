import copy
import sys

def nms(size, stride, outputs, acc_prob=0.0):
    """
    THE FIRST LAYER MUST BE A CONVOLUTIONAL ONE
    Performs Non Maximum Suppression on the outputs dict
    ----------------------------------------------------
    size : size of the square window which produces one prediction
           For convnets, size of the input_space at training (defined in YAML)
    stride : stride when classifying a larger imageize : size of the square
    window which produces one prediction
           For convnets, size of the input_space at training (defined in YAML)
    stride : stride when classifying a larger image,
             is equal to the stride of the first layer,
             is equal to the stride of the first layer
    outputs : dict of output maps indexed by their respective
              scaling factors. These are the predictions on which we apply
              the Non-Maximum Suppression
    """
    local_maxima = {}
    pyramid = copy.copy(outputs)
    for s in pyramid:
        for i in xrange(len(pyramid[s])):
            sys.stdout.write('\r'+str(i)+'/'+str(len(pyramid[s]-1)))
            sys.stdout.flush()
            for j in xrange(len(pyramid[s][i])):
                #print "-"*30
                #for e in pyramid:
                #    print pyramid[e]


                #print "sij",s,i,j,
                s_d = s
                i_d = i
                j_d = j
                center = pyramid[s][i][j]
                if center < acc_prob:
                    pyramid[s][i][j] = 0.0
                    continue
                local_max = True

                # Define the center patch (candidate)
                p_x = i * (1.0/float(s)) * stride
                p_y = j * (1.0/float(s)) * stride
                p = [p_x,
                     p_y,
                     p_x + (1.0/float(s)) * size,
                     p_y + (1.0/float(s)) * size]

                #print "center bbox", p
                #print "center score", center
                #nei = []

                # Test the neighbourhood at all scales
                for s_test in pyramid:
                    if not local_max:
                        break

                    #print s_test
                    i_t = 0
                    j_t = 0
                    # Scale defines a zoom, use 1/s here
                    size_t = float(size)/float(s_test)
                    stride_t = float(stride)/float(s_test)
                    p_test = [0, 0, size_t, size_t]
                    #indices = []
                    maxi, maxj = 0, 0
                    # Scan new scale until overlap
                    while not overlap(p, p_test):

                        if i_t == len(pyramid[s_test]):
                            # No more elements to look at
                            break
                        elif j_t == len(pyramid[s_test][i_t]):
                            # End of a line
                            j_t = 0
                            i_t += 1
                            maxi = max(maxi, i_t)
                        else:
                            j_t += 1
                            maxj = max(maxj, j_t)

                        p_x_t = i_t * stride_t
                        p_y_t = j_t * stride_t
                        p_test = [p_x_t,
                                  p_y_t,
                                  p_x_t + size_t,
                                  p_y_t + size_t]
                        #indices.append(p_test)

                        # Moving on x-axis
                    #if s==s_test:
                        #print "p_test",
                        #print p_test

                    if i_t == len(pyramid[s_test]) or \
                            j_t == len(pyramid[s_test][i_t]):  # no overlap
                        print "_"*20
                        print "no overlap found"
                        print p
                        print "s,  i,  j"
                        print s, i, j
                        print "s_test,i_t,j_t"
                        print s_test, i_t, j_t
                        print maxi, maxj
                        print "len py[s_test],py[s_test][i_t]"
                        print len(pyramid[s_test]),
                        print len(py/ramid[s_test][i_t-1])
                        print "len py[s],py[s][i]"
                        print len(pyramid[s]),
                        print len(pyramid[s][i])
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
                            #nei.append([s_test, i_t + di, j_t + dj])

                            # Compare neighbour and center
                            if (s, i, j) == (s_test, i_t+di, j_t+dj):
                                #print (s,i,j), (s_test, i_t+di, j_t+dj)
                                p_test_var = [p_test[0] + di*stride_t,
                                              p_test[1] + dj*stride_t,
                                              p_test[2] + di*stride_t,
                                              p_test[3] + dj*stride_t]
                                #print "found myself",p_test_var
                            else:
                                neighbour = pyramid[s_test][i_t+di][j_t+dj]

                                if center >= neighbour:
                                    pyramid[s_test][i_t+di][j_t+dj] = 0.0
                                else: # This isn't a local max
                                    #print "isn't a local max"
                                    #print "-----------------"
                                    pyramid[s][i][j] = 0.0
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
                    #print center, [p_x, p_y, int(s*size),
                            #int(s*size)]
                    #print p
                    local_maxima[center] = [p_x, p_y, int(s*size),
                                            int(s*size), center]
                #print (s, i, j)
                #print p
                #print nei
    return pyramid


def overlap(a, b):
    """
    For rectangles defined by 2 points
    """
    return not(a[2]<=b[0] or a[3]<=b[1] or a[0]>=b[2] or a[1]>=b[3])
