#mission3 走雷区
#shymuel 7.26
import cv2 as cv
import math
import numpy as np
from matplotlib import pyplot as plt

cap = cv.VideoCapture('03.mp4')

while True:
    flag, img = cap.read()
    x, y = img.shape[0:2]
    img=cv.resize(img, (int(y/1), int(x/1)))

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))  # 定义矩形结构元素
    img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel, iterations=1) #开运算
    img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel, iterations=1)  # 闭运算1

    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    # xgrad = cv.Sobel(gray, cv.CV_16SC1, 1, 0) #x方向梯度
    # ygrad = cv.Sobel(gray, cv.CV_16SC1, 0, 1) #y方向梯度
    # edge_output = cv.Canny(xgrad, ygrad, 50, 150)
    edge_output = cv.Canny(gray, 60, 180)

    cnts, hierarchy = cv.findContours(edge_output,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)

    for c in cnts:
        area = cv.contourArea(c)
        # print(area)
        if ((area > 500) & (area < 4000)):
            cv.drawContours(img, c, -1, (0, 0, 255), 3)

    # print(cnts)
    cv.imshow('circle', img)
    cv.imshow('Canny', edge_output)
    cv.waitKey(33)
    if (cv.waitKey(1)=='s'):
        while True:
            if (cv.waitKey(1)=='c'):
                break

cv.destroyAllWindows()