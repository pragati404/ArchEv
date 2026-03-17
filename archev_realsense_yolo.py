import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

# -----------------------------
# Load YOLO model
# -----------------------------
model = YOLO("best.pt")

# -----------------------------
# Start RealSense pipeline
# -----------------------------
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

pipeline.start(config)

print("RealSense + YOLO started...")

while True:

    # Get frames
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    if not depth_frame or not color_frame:
        continue

    color = np.asanyarray(color_frame.get_data())

    # -----------------------------
    # YOLO Detection
    # -----------------------------
    results = model(color)

    for box in results[0].boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        label = model.names[class_id]

        # Center of bounding box
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # RealSense depth
        depth = depth_frame.get_distance(cx, cy)

        # Draw bounding box
        cv2.rectangle(color, (x1, y1), (x2, y2), (0,255,0), 2)

        text = f"{label} {confidence:.2f} | {depth:.2f}m"

        cv2.putText(color,
                    text,
                    (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,255,0),
                    2)

    # Display
    cv2.imshow("ArchEV RealSense YOLO", color)

    if cv2.waitKey(1) == 27:
        break

pipeline.stop()
cv2.destroyAllWindows()