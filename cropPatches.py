import cv2 as cv
import os
import argparse
import random
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--cropsize', default = 512, type = int)
parser.add_argument('--overlap', default = 50, type = int)
args = parser.parse_args()

def cropSingle(img, init_args):
    """
    input: a single image, which exceeds 2020Ti's processing size 
    """
    # fetch the parameters of crop
    # assumption: the stirde in width and height are the same 
    size = init_args.cropsize
    overlap = init_args.overlap
    stride = size - overlap

    H, W = img.shape[:2]

    print("Parameters: original size:{}\t{}\nCrop size:{}\nCrop overlap:{}".format(H,W,size,overlap))

    CropPatch = []
    
    # by the cropsize and overlap, all the croping startpoints' coordinates can be calculated
    crop_w_coords = []
    crop_h_coords = []
    
    i = 0
    while (i + size) < W:
        crop_w_coords.append(i)
        i = i + stride
    crop_w_coords.append((W - size))

    j = 0
    while (j + size) < H:
        crop_h_coords.append(j)
        j = j + stride
    crop_h_coords.append((H - size))

    print("find the croping coordinates:{}\t{}".format(crop_w_coords, crop_h_coords))

    #return crop_w_coords, crop_h_coords
    # begin crop process
    for (i, w) in enumerate(crop_w_coords):
        for (j, h) in enumerate(crop_h_coords):
            crop = img[h:h+size, w:w+size]
            CropPatch.append(crop)
            #cv.imwrite(str(i)+'_'+str(j)+'.jpg', crop)
    return crop_w_coords, crop_h_coords, CropPatch

def visualizeCrop(imgs, w_coords, h_coords, size):
    # use this function to see the croping plan for a oversize image
    def getColor():
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        return (r,g,b)

    for w in w_coords:
        for h in h_coords:
            cv.rectangle(imgs, (w,h), (w+size,h+size), getColor(), 8)

    cv.imwrite("./crop_result.jpg", imgs)

def concatPatch(crop_w_coords, crop_h_coords, CropPatch, size):
    """
    input: the crop patch of a oversized image
    output: the whole image after concatenation
    """
    # the concat of two patch is divided into three part:
    # each of independent part (2) + overlap part (1)
    # there is a weighted addition to obtain the overlap part

    # first, get the column part of image
    w_num = len(crop_w_coords)
    h_num = len(crop_h_coords)

    PatchwithColumn = []
    Columns = []
    # each element in the PatchwithColumn is a list of crops that can concat to a column of original image
    # Column is a list of several column of the original image
    def get_overlap(coords, size):
        overlaps = []
        for (idx, coord) in enumerate(coords):
            if (idx+1) == len(coords):
                break
            else:
                overlaps.append(coord + size - coords[idx+1])
        return overlaps

    def concat_to_Column(crop_patch, coords, size):
        """
        crop_patch: a list of crop which can concat to a column of original image
        the process can be interpreted as calculate the UNION SET 
        """
        overlaps = get_overlap(coords, size)
        column_final = crop_patch[0]
        for i in range(len(overlaps)):
            h1 = column_final.shape[0]
            h2 = crop_patch[i+1].shape[0]
            this_overlap = overlaps[i]
            a1 = 0 # start point of upper
            a2 = h1 - this_overlap # independent part of concat element 1
            b1 = 0
            b2 = this_overlap
            sub_union_1 = column_final[a1:a2, :]
            intersection_1 = column_final[a2:h1, :]
            intersection_2 = crop_patch[i+1][b1:b2, :]
            sub_union_2 = crop_patch[i+1][b2:h2, :]
            intersection = 0.5 * intersection_1 + 0.5 * intersection_2
            column_temp = np.concatenate((sub_union_1, intersection, sub_union_2), axis=0)
            column_final = column_temp

        return column_final

    def concat_to_Row(crop_patch, coords, size):
        """
        column_patch: a list of column crop which can concat to the entire original image
        the process can be interpreted as calculate the UNION SET 
        """
        overlaps = get_overlap(coords, size)
        row_final = crop_patch[0]
        for i in range(len(overlaps)):
            w1 = row_final.shape[1]
            w2 = crop_patch[i+1].shape[1]
            this_overlap = overlaps[i]
            a1 = 0 # start point of upper
            a2 = w1 - this_overlap # independent part of concat element 1
            b1 = 0
            b2 = this_overlap
            sub_union_1 = row_final[:, a1:a2]
            intersection_1 = row_final[:, a2:w1]
            intersection_2 = crop_patch[i+1][:, b1:b2]
            sub_union_2 = crop_patch[i+1][:, b2:w2]
            intersection = 0.5 * intersection_1 + 0.5 * intersection_2
            row_temp = np.concatenate((sub_union_1, intersection, sub_union_2), axis=1)
            row_final = row_temp

        return row_final

    for i in range(0, len(CropPatch), h_num):
        column_single = CropPatch[i:i+h_num]
        PatchwithColumn.append(column_single)

    for patchwithcolumn in PatchwithColumn:
        _column = concat_to_Column(patchwithcolumn, crop_h_coords, size)
        Columns.append(_column)

    concat_final = concat_to_Row(Columns, crop_w_coords, size)
    cv.imwrite("./reconstruction.jpg", concat_final)

    return concat_final

    # make a test of column concatenation
    #column_test = concat_to_Column(PatchwithColumn[0], crop_h_coords, size)
    #cv.imwrite("./column_test.jpg", column_test)

def main():
    imgs = cv.imread("./for_users/AIM_IC_t2_test_211_with_holes.png")
    crop_w_coords, crop_h_coords, CropPatch = cropSingle(imgs, args)
    reconstruction = concatPatch(crop_w_coords, crop_h_coords, CropPatch, args.cropsize)
    #visualizeCrop(imgs, coord_w, coord_h, args.cropsize)

if __name__ == '__main__':
    main()