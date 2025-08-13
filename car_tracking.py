import cv2
from ultralytics import YOLO
from collections import defaultdict
import numpy as np

model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture("traffic.mp4")

track_history= defaultdict(lambda: [])
vehicle_classes = [2, 3, 5, 7]

while True:
    ret, frame = cap.read()
    if ret:
        result=model.track(source=frame, persist=True)
        if result[0].boxes.id is not None:
            boxes = result[0].boxes.xywh.cpu()
            track_ids=result[0].boxes.id.int().cpu().tolist()
            cls=result[0].boxes.cls.int().cpu().tolist()
            annotated_frame=result[0].plot()
            for box, id, c in zip(boxes, track_ids,cls):
                if c not in vehicle_classes:
                    continue
                x,y,w,h=box
                track_history[id].append((float(x),float(y)))
                if len(track_history[id])>10:
                    track_history[id]=track_history[id][-7:]
            cv2.imshow("yolo11 tracking", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('a'):
                np.save("track_history.npy", dict(track_history))
                print("Tracking history saved.")
                break
    else:
        break
cap.release()
cv2.destroyAllWindows()

