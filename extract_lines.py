import sys
import cv2
import numpy as np
import os
import pdb


def func():
	#Grab the image
	im = cv2.imread(sys.argv[1])

	#Grab the list of line indices
	fid = open(sys.argv[2],'r')
	fileName = sys.argv[1].split('/')[1]
	# fileName = os.path.join('./line_output/', fileName)

	# pdb.set_trace()

	# outname = sys.argv[1] + '.csv'
	outname = os.path.join('./line_output/', fileName) + '.csv'
	outfile = open(outname, 'w')
	w = im.shape[1]
	lwd = 3

	for i, line in enumerate(fid):
	    linename = os.path.join('./line_output/', fileName.replace('.', '_line'+ str(i) +'.'))
	    a,b = line.strip().split(',')
	    a = int(a); b = int(b)
	    rect = im[(a-1):(b-1),0:w]
	    cv2.imwrite(linename, rect)
	    outfile.write(str(a)+','+str(b)+','+linename+'\n')
	outfile.close()


def main():
	func()

if __name__ == '__main__':
	main()
