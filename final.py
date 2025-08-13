import cv2
from ultralytics import YOLO
from collections import defaultdict
import numpy as np
import csv
from datetime import timedelta

# ------------------ SETTINGS ------------------ #
model = YOLO("yolo11n.pt")  
video_path = "traffic.mp4"
lane_polygons = {
    1: np.load("fast_lane.npy"),  # Lane 1 polygon (n,1,2)
    2: np.load("second_lane.npy"), 
    3: np.load("third_lane.npy"),
}
vehicle_classes = [2, 3, 5, 7]  
output_csv = "vehicle_log.csv"
# ----------------------------------------------- #

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

track_history = defaultdict(lambda: [])
inside_points = defaultdict(list)
lane_counts = defaultdict(set)  # To avoid counting same vehicle twice per lane
frame_num = 0

# CSV setup
with open(output_csv, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Vehicle_ID", "Lane_Number", "Frame_Number", "Timestamp"])

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        # Run YOLO tracking
        results = model.track(source=frame, persist=True)
        annotated_frame = frame.copy()

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            classes = results[0].boxes.cls.int().cpu().tolist()

            for box, track_id, cls_id in zip(boxes, track_ids, classes):
                if cls_id not in vehicle_classes:
                    continue

                x, y, w, h = box
                cx, cy = float(x), float(y)
                track_history[track_id].append((cx, cy))
                if len(track_history[track_id]) > 10:
                    track_history[track_id] = track_history[track_id][-7:]

                # Check each lane
                for lane_num, polygon in lane_polygons.items():
                    result = cv2.pointPolygonTest(polygon, (cx, cy), False)
                    if result >= 0:  # inside or on edge
                        if track_id not in lane_counts[lane_num]:
                            lane_counts[lane_num].add(track_id)  # Count vehicle once
                        if len(inside_points[track_id]) < 2:
                            inside_points[track_id].append((cx, cy))

                        # Write to CSV
                        timestamp = str(timedelta(seconds=frame_num / fps))
                        writer.writerow([track_id, lane_num, frame_num, timestamp])

                # Draw bounding boxes
                cv2.circle(annotated_frame, (int(cx), int(cy)), 4, (0, 255, 0), -1)

            # Draw lane polygons and counts
            for lane_num, polygon in lane_polygons.items():
                cv2.polylines(annotated_frame, [polygon.astype(np.int32)], True, (255, 0, 0), 2)
                count = len(lane_counts[lane_num])
                cv2.putText(annotated_frame, f"Lane {lane_num}: {count}", 
                            (50, 50 + lane_num * 30), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0, 255, 255), 2)

        cv2.imshow("YOLO Lane Tracking", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# Final Summary
print("\nFinal Vehicle Counts Per Lane:")
for lane_num, vehicles in lane_counts.items():
    print(f"Lane {lane_num}: {len(vehicles)} vehicles")
