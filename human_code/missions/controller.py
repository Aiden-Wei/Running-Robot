import threading as thr
import numpy as np
import cv2
# 引入数据库
import human_code.Serial_Servo_Running as cmd

# stream = "http://127.0.0.1:8080/?action=stream?dummy=param.mjpg"
# cap = cv2.VideoCapture(stream)
cap = cv2.VideoCapture(0)

ret, frame_rgb = cap.read()
frame_hsv = np.zeros((480,640,3), dtype=np.unit8)
cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2HSV, frame_hsv)

quit_flag = False # 用于控制子线程退出
mission_cnt = 0 # 任务计数，表示当时的任务状态


# 刷新摄像头数据
def video_cap():
    global frame_rgb, frame_hsv, ret
    global quit_flag
    print("video start")
    while True:
        if frame_rgb is not None and ret:
            ret, frame_rgb = cap.read()
            if quit_flag:
                print('end of sub threat')
                break


color_range_yellow = [([16, 100, 100], [30, 230, 255])]
color_range_red = [()]

mask = None


def color_detect(frame, color_range):
    global mask
    for (lower, upper) in color_range:
        lower = np.array(lower, dtype="uint8")  # 颜色下限
        upper = np.array(upper, dtype="uint8")  # 颜色下限
        mask = cv2.inRange(frame, lower, upper)
        frame_yellow = cv2.bitwise_and(frame, frame, mask=mask)
        cv2.imshow("color_yellow", frame_yellow)


# 默认线程向前走
def default_thread():
    cmd.running_action_group(0, 1)


# 判断哪一个任务
def main():
    global frame_rgb, frame_hsv, ret
    global quit_flag
    video_thr = thr.Thread(target=video_cap)
    video_thr.start()
    while True:
        if frame_rgb is not None and ret:
            cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2HSV, frame_hsv) # 更新hsv图像
            cv2.imshow('rgb', frame_rgb) # 显示图像

            if mission_cnt==0:


            if cv2.waitKey(30) == 27:
                quit_flag = True
                break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
