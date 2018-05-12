import cv2
import sys
import numpy as np
import os
import networkx as nx
import pdb


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


#Let's keep angles in -45 to 45
def rectify(x):
    if x < -45:
        x = x + 90
    elif x > 45:
        x = x - 90
    return(float(x))

def vec_tv(x):
    return(np.sum(np.abs(np.diff(x))))

def guess_rot(rect, im3a, imtext):
    ratio_cutoff = 4.0
    min_cutoff = 20
    max_cutoff = 800
    area_min = 10

    chunk = crop_minAreaRect(imtext, rect, im3a)
    ret, thresh = cv2.threshold(chunk,127,255,0)
    comp = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    # pdb.set_trace()
    for i in range(len(comp[1])):
        box = cv2.minAreaRect(comp[1][i])
        area = np.prod(box[1])
        if area < area_min:
            continue
        if box[2] == 0.0:
            continue
        ratio = max(box[1][0]/float(box[1][1]), box[1][1]/float(box[1][0]))
        maxlen = max(2*box[1][0],2*box[1][1])

        #box = (box[0], (box[1][0], box[1][1]), box[2])
        if (ratio > ratio_cutoff) and (maxlen > min_cutoff) and (maxlen < max_cutoff):
            cv2.fillPoly(chunk, [comp[1][i]], 128)
            boxes.append((area, box))


    boxes.sort(key=lambda x: x[0], reverse=True)

    angles = np.array([rectify(b[1][2]) for b in boxes])
    return(np.mean(angles))


def tv_tune(imtext, rect, img, guess=0.0, search_width=2):
    #This one takes in the actual text picture, for now.
    ratio_cutoff = 4.0
    min_cutoff = 20
    max_cutoff = 800
    area_min = 10

    chunk = crop_minAreaRect(imtext, rect, img)
    ret, thresh = cv2.threshold(chunk,127,255,0)
    grid = np.arange(guess - search_width, guess+search_width, 0.05)
    tvs = np.zeros(len(grid))
    for i, ang in enumerate(grid):
        rotmat = cv2.getRotationMatrix2D((chunk.shape[1]/2, chunk.shape[0]/2),
                ang, 1)
        siderot = cv2.warpAffine(chunk, rotmat, (chunk.shape[1], chunk.shape[0]),
            None,cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, (255,255,255))
        # siderotBW = cv2.cvtColor(siderot, cv2.COLOR_RGB2GRAY)
        sums = np.sum(thresh, axis=1, dtype=np.float64) # call on siderotBW
        #print(thresh[0,0])
        #print(np.sum(sums))
        tvs[i] = vec_tv(sums)

    # pdb.set_trace()
    try:
        ans = grid[np.argmax(tvs)]
    except:
        ans = 0.0
    return ans


 
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

    rot1 = guess_rot(boxes[0][1], org_img, imtext) # Init
    # pdb.set_trace()
    print('rotation 1 '+str(rot1))
    adjrot1 = tv_tune(imtext, boxes[0][1], org_img, guess=rot1, search_width=2)
    print('rotation 1 adj '+str(adjrot1))
    rot1 = adjrot1

    rot2 = guess_rot(boxes[1][1], org_img, imtext)
    print('rotation 2 '+str(rot2))
    adjrot2 = tv_tune(imtext, boxes[0][1], org_img, guess=rot2, search_width=2)
    print('rotation 2 adj '+str(adjrot2))
    rot2 = adjrot2

    rotmat1 = cv2.getRotationMatrix2D((side1.shape[1]/2, side1.shape[0]/2), rot1, 1)
    rotmat2 = cv2.getRotationMatrix2D((side2.shape[1]/2, side2.shape[0]/2), rot2, 1)
    side1rot = cv2.warpAffine(side1, rotmat1, (side1.shape[1], side1.shape[0]),
            None,cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, (255,255,255))
    side2rot = cv2.warpAffine(side2, rotmat2, (side2.shape[1], side2.shape[0]),
            None,cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, (255,255,255))

    doc_name = image_file.split('/')[1].split('.')[0]

    output_path = os.path.join('./step2_rot/', doc_name +'._step2')

    print(cv2.imwrite(os.path.join('./step2_rot/', doc_name +'._step2_1.jpg'), side1))
    print(cv2.imwrite(os.path.join('./step2_rot/', doc_name +'._step2_2.jpg'), side2))
    print(cv2.imwrite(os.path.join('./step2_rot/', doc_name +'._step2_1r.jpg'), side1rot))
    print(cv2.imwrite(os.path.join('./step2_rot/', doc_name +'._step2_2r.jpg'), side2rot))



if __name__ == "__main__":
    main()

    