#mission2 上台阶
#shymuel 7.26
#摄像头跟随台阶移动

import cv2 as cv
import numpy as np
import pickle
import matplotlib.pyplot as plt
import time
import math
import lsc

pitch = 1500 #初始位置为中间位置
yaw = 1500
lsc.MoveServo(6, 1500,1000)  #让摄像头云台的两个舵机都转动到中间位置
lsc.MoveServo(7, 1500,1000)

def calc_bottom(rectangle):
    print(rectangle)
    return rectangle[0][1] + 0.5*(rectangle[1][0]* math.sin(math.fabs(rectangle[2])) +rectangle[1][1]*math.cos(math.fabs(rectangle[2])))

def adjustCamera(rect, pitch ,yaw):
    xc = True #xc代表是否调整X方向
    yc = True #yc代表是否调整y方向
    centerX = rect[0][1]
    centerY = rect[0][0]

    # 范围为0-640
    if centerX > 620:
        pitch = pitch - 50  # 根据中心位置确定要调整的舵机角度,距离中心越大调整幅度就越大
    elif centerX > 540:
        pitch = pitch - 30
    elif centerX > 380:
        pitch = pitch - 15
    elif centerX > 325:
        pitch = pitch - 2
    elif centerX > 315:
        xc = False  # 在中心范围内，不用调整舵机
    elif centerX > 260:
        pitch = pitch + 2
    elif centerX > 100:
        pitch = pitch + 15
    elif centerX > 20:
        pitch = pitch + 30
    elif centerX > 0:
        pitch = pitch + 50
    else:
        xc = False  # 屏幕中没有球， 不调整舵机

    # 范围为0-480
    if centerY > 450:
        yaw = yaw + 40  # 根据中心位置确定要调整的舵机角度,距离中心越大调整幅度就越大
    elif centerY > 380:
        yaw = yaw + 25
    elif centerY > 310:
        yaw = yaw + 15
    elif centerY > 245:
        yaw = yaw + 2
    elif centerY > 235:
        yc = False
    elif centerY > 170:
        yaw = yaw - 2
    elif centerY > 100:
        yaw = yaw - 15
    elif centerY > 30:
        yaw = yaw - 25
    elif centerY > 0:
        yaw = yaw - 40
    else:
        yc = False

    if xc is True:  # 舵机角度被改变
        pitch = pitch if pitch <= 2500 else 2500  # 限制舵机角度的最大最小值
        pitch = pitch if pitch >= 500 else 500
        lsc.MoveServo(6, pitch, 50)  # 让舵机转到新的角度去

    if yc is True:
        yaw = yaw if yaw <= 2500 else 2500
        yaw = yaw if yaw >= 500 else 500
        lsc.MoveServo(7, yaw, 50)

    return pitch, yaw


def findStep(frame):
    HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)  # 把BGR图像转换为HSV格式
    mask1 = cv.inRange(HSV, np.array([0, 43, 46]), np.array([10, 255, 255]))
    mask2 = cv.inRange(HSV, np.array([156, 43, 46]), np.array([180, 255, 255]))
    mask2 = cv.bitwise_or(mask1, mask2, mask=None)

    #print(mask2.shape)  #480*640

    cnts, hierarchy = cv.findContours(mask2,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)

    if cnts == []:
        return 0
    c = max(cnts, key=cv.contourArea)
    return cv.minAreaRect(c)

cap = cv.VideoCapture(0)

while cap.isOpened():
    grabbed, frame = cap.read()
    # print(frame.shape)
    rect = findStep(frame) #最小外接方框
    pitch, yaw = adjustCamera(rect, pitch, yaw) #调整摄像头

    if (rect!=0):
        bottom = calc_bottom(rect)
    print(bottom)

    if rect == 0:
        continue

    box = cv.boxPoints(rect)
    box = np.int0(box)
    cv.drawContours(frame, [box], -1, (0, 255, 0), 2)
    cv.line(frame, (box[1, 0], box[1, 1]), (box[3, 0], box[3, 1]), 2, 2)
    cv.line(frame, (box[0, 0], box[0, 1]), (box[2, 0], box[2, 1]), 2, 2)
    cv.imshow("capture", frame)

    if cv.waitKey(33) == 27:
        break

cap.release()
cv.destroyWindow("capture")





