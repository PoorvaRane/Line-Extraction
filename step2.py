import cv2
import sys
import numpy as np
import os
import networkx as nx


def crop_minAreaRect(img, rect, org_img):
    flip=False
    angle = rect[2]
    if angle < -45:
        angle = angle+90
        flip=True
    if angle > 45:
        angle = angle-90
        flip=True
    rows,cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    img_rot = cv2.warpAffine(img,M,(cols,rows))

    # rotate bounding box
    rect0 = (rect[0], rect[1], 0.0)
    if flip:
        rect0 = (rect[0], (rect[1][1],rect[1][0]), 0.0)
    box = cv2.boxPoints(rect0)
    pts = np.int0(cv2.transform(np.array([box]), M))[0]    
    pts[pts < 0] = 0

    # crop
    # img_crop = img_rot[pts[1][1]:pts[0][1], pts[1][0]:pts[2][0]]

    img_crop = org_img[pts[1][1]:pts[0][1], pts[1][0]:pts[2][0]]

    return img_crop


 
def main():
    image_file = sys.argv[1]
    im = cv2.imread(image_file, 0)
    imtext = cv2.imread(image_file)
    imc = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    ret, thresh = cv2.threshold(im,127,255,0)
    comp = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    org_image_file = sys.argv[2]
    org_img = cv2.imread(org_image_file, 0)

    mergethresh = 0.993
    pad = 20

    # New version using contours
    G = nx.Graph()
    for i in range(len(comp[1])):
        x1,y1,w1,h1 = cv2.boundingRect(comp[1][i])
        G.add_node(i)
        for j in range(i+1,len(comp[1])):
            x2,y2,w2,h2 = cv2.boundingRect(comp[1][j])
            overlap = min(x1+w1,x2+w2)-max(x1,x2)
            base = min(w1,w2)
            if (overlap/float(base)) > mergethresh:
                # print i,j,overlap, base
                G.add_edge(i,j)

    blocks = [a for a in nx.connected_components(G)]

    boxes = []
    for blk in blocks:
        pts = np.concatenate([comp[1][j] for j in blk],axis=0)
        box = cv2.minAreaRect(pts)
        area = np.prod(box[1])
        # print area
        box = (box[0], (box[1][0]+2*pad, box[1][1]+2*pad), box[2])
        boxes.append((area, box))

    boxes.sort(key=lambda x: x[0], reverse=True)

    side1 = crop_minAreaRect(imtext, boxes[0][1], org_img)
    side2 = crop_minAreaRect(imtext, boxes[1][1], org_img)

    doc_name = image_file.split('/')[1].split('.')[0]

    output_path = os.path.join('./step2_output/', doc_name +'._step2')

    print(cv2.imwrite(os.path.join('./step2_output/', doc_name +'._step2_1.jpg'), side1))
    print(cv2.imwrite(os.path.join('./step2_output/', doc_name +'._step2_2.jpg'), side2))



if __name__ == "__main__":
    main()

    