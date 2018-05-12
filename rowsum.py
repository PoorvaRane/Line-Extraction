import cv2
import numpy as np
import sys


def func():
	im = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
	print(im.shape)

	im = 1-im/255
	sums = np.sum(im, axis=1)
	# print(len(sums))
	# print(sums.shape)
	# print(np.max(im, axis=1))

	fname = sys.argv[1].split('/')[1].split('.')[0] + '.csv'
	fout = open(fname,'w')
	for x in sums:
	    fout.write(str(x)+'\n')
	fout.close()


def main():
	func()

if __name__ == '__main__':
	main()