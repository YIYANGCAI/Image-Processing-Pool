"""
Current text detection task is really dufficult 
because traditional bboxes format cannot precisely 
obtain the text's real location.
Polygons are now a fashion type to locate the text
This function find a circumscribe rectangle for a
polygon and crop it out
"""
from skimage import io
import os
from math import *
import cv2 as cv
import numpy as np

def rotateImage(img,degree,pt1,pt2,pt3,pt4):
    '''
    input: original img, the text area with rotation degree, four points' coordinates
    output: the segment of the text area, with horizontal view
    '''
    height,width=img.shape[:2]
    heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))
    matRotation=cv.getRotationMatrix2D((width/2,height/2),degree,1)
    matRotation[0, 2] += (widthNew - width) / 2
    matRotation[1, 2] += (heightNew - height) / 2
    imgRotation = cv.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
    pt1 = list(pt1)
    pt3 = list(pt3)
    [[pt1[0]], [pt1[1]]] = np.dot(matRotation, np.array([[pt1[0]], [pt1[1]], [1]]))
    [[pt3[0]], [pt3[1]]] = np.dot(matRotation, np.array([[pt3[0]], [pt3[1]], [1]]))
    #print(imgRotation)
    print(pt1[1])
    print(pt3[1])
    print(pt1[0])
    print(pt3[0])
    # for the style transfer we need to add some margin
    margin_1 = int((int(pt3[0]) - int(pt1[0]))/2)
    margin_2 = int((int(pt1[1]) - int(pt3[1]))/2)
    margin = min([margin_1, margin_2])
    imgOut=imgRotation[int(pt3[1])-margin:int(pt1[1])+margin, int(pt1[0])-margin:int(pt3[0])+margin]
    if degree <= -45:
        imgOut = np.rot90(imgOut)
    return imgOut

def findCircumLine(img, pointsList):
    """
    pointsList records the coordinates for a polygon
    format: list of tuple
    """
    rect = cv.minAreaRect(np.array(pointsList))
    new_box = cv.boxPoints(rect)
    new_box = np.int0(new_box)                    
    print("normalized box coordinates:\n{}".format(new_box))
    print("angle of rotation:\n{}".format(rect[2]))
    p1 = (new_box[0,0], new_box[0,1])
    p2 = (new_box[1,0], new_box[1,1])
    p3 = (new_box[2,0], new_box[2,1])
    p4 = (new_box[3,0], new_box[3,1])
    angle = rect[2]
    extracted_area = rotateImage(img, angle, p1, p2, p3, p4)
    return extracted_area

def visualizeRotate(img, pointsList):
    pass
