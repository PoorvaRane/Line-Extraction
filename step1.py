import cv2
import numpy as np
import sys
import os


def main():
	image_file = sys.argv[1]
	lines = int(sys.argv[2]) #approx = 33 for smaple.tif

	img = cv2.imread(image_file, 0)
	height, width = img.shape

	linespace = height/lines*13/16

	s1 = height/450
	s2 = height/450
	s3 = height/750
	sq = linespace*3/2 
	cut = height/float(11)
	vcut = linespace
	hcut = linespace

	print s1, s2, s3, sq, cut, vcut, hcut

	blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, np.ones((int(s1), int(s1)), np.uint8))
	# print cv2.imwrite('1_python.tiff', blackhat)

	img = cv2.dilate(blackhat, np.ones((int(s1), int(s1)), np.uint8), iterations = 4)
	# print cv2.imwrite('2_python.tiff', img)

	tmp1 = cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((1, vcut), np.uint8))
	# print cv2.imwrite('tmp1.jpg',tmp1)
	tmp2 = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((hcut, 1), np.uint8))
	# print cv2.imwrite('tmp2.jpg',tmp2)
	rectangle = cv2.morphologyEx(tmp2, cv2.MORPH_RECT, np.ones((30,10), np.uint8))
	doc_name = image_file.split('/')
	doc_name = doc_name[1].split('.')
	output_path = os.path.join('./step1_output/', doc_name[0] +'._step1.jpg')
	print output_path
	print cv2.imwrite(output_path, rectangle)


if __name__ == "__main__":
	main()
