import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

# -----------------------------
# Load YOLO model
# -----------------------------
model = YOLO("best.pt")

# Confidence threshold
CONF_THRESHOLD = 0.3

# -----------------------------
# Start RealSense pipeline
# -----------------------------
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

pipeline.start(config)

print("ArchEV RealSense + YOLO started")

while True:

    # Wait for frames
    frames = pipeline.wait_for_frames()

    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    if not depth_frame or not color_frame:
        continue

    # Convert to numpy arrays
    color = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    # -----------------------------
    # Depth map visualization
    # -----------------------------
    depth_scaled = cv2.convertScaleAbs(depth_image, alpha=0.1)

    depth_colormap = cv2.applyColorMap(
        depth_scaled,
        cv2.COLORMAP_JET
    )

    # -----------------------------
    # YOLO Detection
    # -----------------------------
    results = model(color, imgsz=1280)

    for box in results[0].boxes:

        confidence = float(box.conf[0])

        # Filter weak detections
        if confidence < CONF_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        class_id = int(box.cls[0])
        label = model.names[class_id]

        # Bounding box center
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # Get real depth (meters)
        depth = depth_frame.get_distance(cx, cy)

        # Draw bounding box
        cv2.rectangle(color, (x1, y1), (x2, y2), (0,255,0), 2)

        # Draw center point
        cv2.circle(color, (cx, cy), 5, (0,0,255), -1)

        # Display text
        text = f"{label} {confidence:.2f} | {depth:.2f} m"

        cv2.putText(
            color,
            text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,255,0),
            2
        )

    # -----------------------------
    # Display windows
    # -----------------------------
    cv2.imshow("ArchEV YOLO Detection", color)
    cv2.imshow("ArchEV Depth Map", depth_colormap)

    if cv2.waitKey(1) == 27:
        break

pipeline.stop()
cv2.destroyAllWindows()