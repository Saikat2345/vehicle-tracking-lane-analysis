import numpy as np
import cv2
data=np.load("fast_lane.npy")
data2=np.load("second_lane.npy")
data3=np.load("third_lane.npy")
print(data)
cap=cv2.VideoCapture("traffic.mp4")
res,frame=cap.read()
cap.release()
if res:
    cv2.polylines(frame, [data], isClosed=True, color=(0,255,0), thickness=2)
    cv2.polylines(frame, [data2], isClosed=True, color=(255,0,0), thickness=2)
    cv2.polylines(frame, [data3], isClosed=True, color=(255,0,0), thickness=2)

    cv2.imshow("lane",frame)

while True:
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
cv2.destroyAllWindows()        