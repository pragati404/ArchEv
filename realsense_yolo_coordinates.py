# Optimized RealSense + YOLO + Depth Detection

import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

# -----------------------------
# Load YOLO model
# -----------------------------
model = YOLO("best.pt")

# Confidence threshold
CONF_THRESHOLD = 0.5

# -----------------------------
# Start RealSense pipeline
# -----------------------------
pipeline = rs.pipeline()
config = rs.config()

# Lower resolution = faster performance
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

pipeline.start(config)

# Align depth to color frame
align = rs.align(rs.stream.color)

# Frame skipping for faster FPS
frame_skip = 2
frame_count = 0

print("ArchEV RealSense + YOLO started")

try:
    while True:

        # -----------------------------
        # Frame skipping
        # -----------------------------
        frame_count += 1

        if frame_count % frame_skip != 0:
            continue

        # -----------------------------
        # Wait for frames
        # -----------------------------
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)

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
        depth_scaled = cv2.convertScaleAbs(depth_image, alpha=0.08)

        depth_colormap = cv2.applyColorMap(
            depth_scaled,
            cv2.COLORMAP_JET
        )

        # -----------------------------
        # YOLO Detection
        # -----------------------------
        results = model(
            color,
            imgsz=640,
            conf=CONF_THRESHOLD,
            verbose=False
        )

        for box in results[0].boxes:

            confidence = float(box.conf[0])

            if confidence < CONF_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            class_id = int(box.cls[0])
            label = model.names[class_id]

            # Bounding box center
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Prevent out-of-bounds indexing
            h, w = depth_image.shape

            cx = np.clip(cx, 2, w - 3)
            cy = np.clip(cy, 2, h - 3)

            # -----------------------------
            # Stable depth averaging
            # -----------------------------
            region = depth_image[cy-2:cy+3, cx-2:cx+3]

            valid_depths = region[region > 0]

            if len(valid_depths) > 0:
                depth = np.mean(valid_depths) * 0.001
            else:
                depth = 0

            # -----------------------------
            # Convert pixel to real-world XYZ coordinates
            # -----------------------------
            intrinsics = color_frame.profile.as_video_stream_profile().intrinsics

            point_3d = rs.rs2_deproject_pixel_to_point(
                intrinsics,
                [cx, cy],
                depth
            )

            X_real = point_3d[0]
            Y_real = point_3d[1]
            Z_real = point_3d[2]

            # -----------------------------
            # Draw Bounding Box
            # -----------------------------
            cv2.rectangle(
                color,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Center point
            cv2.circle(
                color,
                (cx, cy),
                5,
                (0, 0, 255),
                -1
            )

            # Object coordinates (real-world meters)
            coord_text = (
                f"X:{X_real:.2f}m Y:{Y_real:.2f}m Z:{Z_real:.2f}m"
            )

            # Detection label
            detection_text = f"{label} {confidence:.2f} | Depth:{depth:.2f}m"

            # Draw label above bounding box
            cv2.putText(
                color,
                detection_text,
                (x1, y1 - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            # Draw XYZ coordinates below bounding box
            cv2.putText(
                color,
                coord_text,
                (x1, y2 + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                2
            )

            # Print XYZ coordinates to terminal
            print(
                f"Detected: {label} | Confidence: {confidence:.2f} | X: {X_real:.3f}m | Y: {Y_real:.3f}m | Z: {Z_real:.3f}m | Depth: {depth:.2f}m"
            )
            print(
                f"Detected: {label} | Confidence: {confidence:.2f} | X: {cx} | Y: {cy} | Z: {depth:.2f} m"
            )
            print(
                f"Detected: {label} | Confidence: {confidence:.2f} | X: {cx} | Y: {cy} | Depth: {depth:.2f} m"
            )

        # -----------------------------
        # Display windows
        # -----------------------------
        cv2.imshow("ArchEV YOLO Detection", color)
        cv2.imshow("ArchEV Depth Map", depth_colormap)

        # ESC key to quit
        if cv2.waitKey(1) == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

