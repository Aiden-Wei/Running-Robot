#mission4 跨挡板
#by sbz

import cv2
import time
import numpy as np
import math

#定义一些参数
ori_width  =  int(4*160)#原始图像640x480
ori_height =  int(3*160)
resize_width   = int(4*20)#处理图像时缩小为80x60,加快处理速度，谨慎修改！
resize_height  = int(3*20)
line_color     = (255, 0, 0)#图像显示时，画出的线框颜色
line_thickness = 2         #图像显示时，画出的线框的粗细
roi = [ # [ROI, weight]
        (0,  40,  0, 160, 0.5),
        (40, 80,  0, 160, 0.3),
        (80, 120,  0, 160, 0.2)
       ]

def edge_detection(orgimage, r_w = resize_width, r_h = resize_height, r = roi, l_c = line_color, l_t = line_thickness):
    # 缩小图像
    orgframe = cv2.resize(orgimage, (r_w, r_h), interpolation=cv2.INTER_LINEAR)
    orgframe = cv2.cvtColor(orgframe, cv2.COLOR_BGR2GRAY)  # 转化为灰度图像
    orgframe = cv2.GaussianBlur(orgframe, (3, 3), 0)  # 高斯模糊，去噪

    # 边缘检测测试
    orgframe_sobel = cv2.Sobel(orgframe, cv2.CV_16S, 0, 2)#检测水平边缘
    orgframe_sobel = cv2.convertScaleAbs(orgframe_sobel)
    cv2.imshow("orgframe_sobel", orgframe_sobel)
    # cv2.waitKey(30)

#导入视频流
cap = cv2.VideoCapture('C:\\Users\\DELL\\Desktop\\赛道视频_2019_07_27\\04跨挡板.mp4')

# 设置蓝色色的范围
color_range_blue = [([78, 43, 46], [130, 255, 255])]

mask = None
def color_detect(frame, color_range): #hsv的图像
    global mask
    for (lower, upper) in color_range:
        lower = np.array(lower, dtype="uint8")  # 颜色下限
        upper = np.array(upper, dtype="uint8")  # 颜色下限

        mask = cv2.inRange(frame, lower, upper)
        frame_blue = cv2.bitwise_and(frame, frame, mask=mask)

        cv2.imshow("color_blue", frame_blue)

def return_color(event, x, y, flags, param):  #看颜色
    global frame_hsv
    global cal_h, cal_s, cal_v
    if event == cv2.EVENT_LBUTTONDOWN:
        print((x,y))
        print(frame_hsv[y,x])

def getAreaMaxContour(contours, area=1): #找到最大面积
    contour_area_max = 0
    area_max_contour = None

    for c in contours:
        contour_area_temp = math.fabs(cv2.contourArea(c))
        if contour_area_temp > contour_area_max:
            contour_area_max = contour_area_temp
            if contour_area_temp > area:  # 面积大于1
                area_max_contour = c
    return area_max_contour

def getAreaSecondContour(contours, max, area=1): #找到第二大颜色
    contour_area_second = 0
    area_second_contour = None

    for c in contours:
        contour_area_temps = math.fabs(cv2.contourArea(c))
        contour_area_max = math.fabs(cv2.contourArea(max))
        if contour_area_max > contour_area_temps > contour_area_second:
            contour_area_second = contour_area_temps
            if contour_area_temps > area:  # 面积大于1
                area_second_contour = c
    return area_second_contour

if __name__ == "__main__":
    while True:
        ret, frames = cap.read()
        frame_hsv = frames.copy()
        cv2.cvtColor(frames, cv2.COLOR_BGR2HSV, frame_hsv)
        if frames is not None and ret:
            # edge_detection(frames, 160, 120)
            color_detect(frame_hsv, color_range_blue)

            cv2.setMouseCallback('frame', return_color)
        # 画出矩形轮廓
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)  # 找出所有轮廓
            cnt_large = getAreaMaxContour(cnts, area=10)  # 找到最大面积的轮廓
            cnt_second = getAreaSecondContour(cnts, cnt_large, area=10) # 找到第二大面积的轮廓
            if cnt_second is not None:
                rect = cv2.minAreaRect(cnt_second)  # 最小外接矩形
                box = np.int0(cv2.boxPoints(rect))  # 最小外接矩形的四个顶点
                cv2.drawContours(frames, [box], -1, (0, 0, 255, 255), 2)  # 画出四个点组成的矩形
                cv2.line(frames, (box[1, 0], box[1, 1]), (box[3, 0], box[3, 1]), line_color, line_thickness)
                cv2.line(frames, (box[0, 0], box[0, 1]), (box[2, 0], box[2, 1]), line_color, line_thickness)

            cv2.imshow("frame", frames)
            if  cv2.waitKey(30)==32:#spacce键暂停
                cv2.waitKey(-1)
            if  cv2.waitKey(30)==27:
                break
            '''if cnt_large is None:
                break'''
    cv2.destroyAllWindows()