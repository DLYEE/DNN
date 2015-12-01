#!/usr/bin/env python
# from http://martin-thoma.com/word-error-rate-calculation/ 
 
import numpy
import re
# import sys

def editDistance(label, solution):
    # r = sys.argv[1].split();
    # print r
    # h = sys.argv[2].split();
    # print h
    f1 = open(label, "r+")
    r = []
    for line in open(label):
        line = f1.readline()
        if line[:2] == 'id':
            continue
        else:
            phone = re.split(",|\n", line)
            phone.pop()
            for i in phone[1]:
                r.append(i)
    f1.close()
    print r
    f2 = open(solution, "r+")
    h = []
    for line in open(solution):
        line = f2.readline()
        if line[:2] == 'id':
            continue
        else:
            phone = re.split(",|\n", line)
            phone.pop()
            for i in phone[1]:
                h.append(i)
    f2.close()
    print h

    d = numpy.zeros((len(r)+1)*(len(h)+1), dtype=numpy.uint8)
    d = d.reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        for j in range(len(h)+1):
                if i == 0:
                        d[0][j] = j
                elif j == 0:
                    d[i][0] = i

    # computation
    for i in range(1, len(r)+1):
            for j in range(1, len(h)+1):
                    if r[i-1] == h[j-1]:
                        d[i][j] = d[i-1][j-1]
                    else:
                        substitution = d[i-1][j-1] + 1
                        insertion    = d[i][j-1] + 1
                        deletion     = d[i-1][j] + 1
                        d[i][j] = min(substitution, insertion, deletion)

    #edit distance
    print (d[len(r)][len(h)], len(r))

editDistance("testLabel", "testSolution")
#error rate
#print float(d[len(r)][len(h)])/float(len(r))

