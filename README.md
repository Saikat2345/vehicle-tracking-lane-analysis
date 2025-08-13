# 🚦 Traffic Flow Analysis

This project was developed for **Skytel Tele Services Pvt. Ltd.** to analyze traffic flow using computer vision.  
The system detects, tracks, and counts vehicles in **three distinct lanes** from a given traffic video,  
exporting results to a CSV file and overlaying real-time counts on the processed video.

---

## 📌 Objective
The goal is to **accurately detect and count vehicles** in each lane, ensuring no duplicate counts,  
and provide both visual and structured data outputs for traffic flow monitoring.

---

## 🛠 Features
- **Vehicle Detection**: Uses a pre-trained COCO model (YOLO/SSD) for object detection.
- **Lane Definition & Counting**: Three lanes manually defined; counts maintained separately.
- **Vehicle Tracking**: Integrated tracking algorithm (e.g., SORT/DeepSORT) to prevent duplicate counts.
- **Real-time Processing**: Optimized for near real-time performance on standard hardware.
- **CSV Export**: Outputs `Vehicle ID`, `Lane Number`, `Frame Count`, and `Timestamp`.
- **Visual Overlay**: Displays lane boundaries and live vehicle counts per lane.
- **Summary Report**: Shows total vehicle count per lane after processing.
