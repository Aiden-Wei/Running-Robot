#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import cv2
from human_code.line_patrol import resize_width, resize_height, roi, line_color, line_thickness

def edge_detection(orgimage, r_w = resize_width, r_h = resize_height, r = roi, l_c = line_color, l_t = line_thickness):
    orgframe = cv2.resize(orgimage, (r_w, r_h), interpolation=cv2.INTER_LINEAR)
    orgframe = cv2.cvtColor(orgframe, cv2.COLOR_BGR2GRAY)  # 转化为灰度图像
    orgframe = cv2.GaussianBlur(orgframe, (3, 3), 0)  # 高斯模糊，去噪

    # 边缘检测测试
    orgframe_sobel = cv2.Sobel(orgframe, cv2.CV_16S, 0, 1)
    orgframe_sobel = cv2.convertScaleAbs(orgframe_sobel)
    cv2.imshow("orgframe_sobel", orgframe_sobel)
    # cv2.waitKey(30)
