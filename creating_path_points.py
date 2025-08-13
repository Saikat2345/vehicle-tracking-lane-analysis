# import numpy as np
# data = np.load("track_history.npy", allow_pickle=True).item()
# print(data[1])

import cv2
import numpy as np

# Load video
cap = cv2.VideoCapture("traffic.mp4")  # Make sure you've downloaded it
ret, frame = cap.read()
cap.release()

if not ret:
    print("Failed to read video")
    exit()

points = []

def click_event(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:  # Left click
        points.append((x, y))
        print(f"Point selected: ({x}, {y})")
        cv2.circle(param, (x, y), 5, (0, 0, 255), -1)  # Draw a small red dot
        cv2.imshow("Frame", param)
temp_frame = frame.copy()
cv2.imshow("Frame", temp_frame)
cv2.setMouseCallback("Frame", click_event, temp_frame)

print("Click to select points, press 'q' when done.")

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
countour=np.array(points,dtype=np.int32).reshape((-1,1,2))
# np.save("first_lane",countour)
# np.save("second_lane.npy", countour)
np.save("third_lane.npy",countour)
cv2.destroyAllWindows()
print("Selected Points:", countour)



