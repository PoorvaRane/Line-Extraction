import sys
import cv2
import numpy as np
import pdb
import os

def main():
    #Grab the image
    im = cv2.imread(sys.argv[1])

    #Grab the list of line indices
    fid = open(sys.argv[2],'r')
    lines = [int(x.strip()) for x in fid]
    fid.close()

    fname = sys.argv[1].replace('.', '_lined.')
    fname = os.path.join('./line_output/', fname.split('/')[1])
    w = im.shape[1]
    lwd = 3

    for l in lines:
        cv2.line(im, (0, l), (w,l), (0,0,255),lwd)

    cv2.imwrite(fname,im)

if __name__ == '__main__':
    main()

