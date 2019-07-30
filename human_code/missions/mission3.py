#mission3 走雷区
#shymuel 7.26
import cv2 as cv
import math
import numpy as np
from matplotlib import pyplot as plt

def swap(t1, t2):
    t2, t1 = t1, t2
    return

cap = cv.VideoCapture('03.mp4')

while True:
    flag, img = cap.read()
    x, y = img.shape[0:2]
    img=cv.resize(img, (int(y/1), int(x/1)))

    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    # xgrad = cv.Sobel(gray, cv.CV_16SC1, 1, 0) #x方向梯度
    # ygrad = cv.Sobel(gray, cv.CV_16SC1, 0, 1) #y方向梯度
    # edge_output = cv.Canny(xgrad, ygrad, 50, 150)
    edge_output = cv.Canny(gray, 80, 255)

    cnts, hierarchy = cv.findContours(edge_output,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)

    for i in cnts.size():
        area = cv.contourArea(cnts[i])
        # print(area)
        swap(cnts[i], cnts[cnts.size() - 1]);
        cnts.pop_back();

    cv.drawContours(img,cnts,-1,(0,0,255),3)
    # print(cnts)
    cv.imshow('circle', img)
    cv.imshow('Canny', edge_output)
    cv.waitKey(33)
    if (cv.waitKey(1)=='s'):
        while True:
            if (cv.waitKey(1)=='c'):
                break

cv.destroyAllWindows()