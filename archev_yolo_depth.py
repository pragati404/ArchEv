import cv2
import torch
import numpy as np
import open3d as o3d
from ultralytics import YOLO


# Load YOLO model

model = YOLO("best.pt")


# Load MiDaS depth model

midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.eval()

transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = transforms.small_transform


# Webcam

cap = cv2.VideoCapture(0)


# Open3D viewer

vis = o3d.visualization.Visualizer()
vis.create_window("ArchEV 3D Point Cloud", width=800, height=600)

pcd = o3d.geometry.PointCloud()

while True:

    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    cx = w/2
    cy = h/2

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # -------------------------
    # YOLO detection
    # -------------------------
    results = model(frame)

    center_x, center_y = None, None

    for box in results[0].boxes.xyxy:

        x1,y1,x2,y2 = map(int,box)

        center_x = int((x1+x2)/2)
        center_y = int((y1+y2)/2)

        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

    # -------------------------
    # Depth estimation
    # -------------------------
    input_batch = transform(rgb)

    with torch.no_grad():

        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth = prediction.cpu().numpy()

    depth = cv2.GaussianBlur(depth,(9,9),0)

    depth_vis = cv2.normalize(depth,None,0,255,cv2.NORM_MINMAX)
    depth_vis = depth_vis.astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_vis,cv2.COLORMAP_INFERNO)

    # -------------------------
    # 3D coordinate from YOLO box
    # -------------------------
    if center_x is not None:

        Z = depth[center_y,center_x]

        X = (center_x - cx)*Z
        Y = -(center_y - cy)*Z

        cv2.circle(frame,(center_x,center_y),5,(0,0,255),-1)

        cv2.putText(frame,
                    f"X:{X:.2f} Y:{Y:.2f} Z:{Z:.2f}",
                    (center_x,center_y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,255,0),
                    2)

    # -------------------------
    # Create point cloud
    # -------------------------
    step = 4

    xx,yy = np.meshgrid(np.arange(0,w,step),
                        np.arange(0,h,step))

    z = depth[0:h:step,0:w:step].flatten()

    x3 = (xx.flatten()-cx)*z
    y3 = -(yy.flatten()-cy)*z

    points = np.vstack((x3,y3,z)).transpose()

    colors = rgb[0:h:step,0:w:step].reshape(-1,3)/255.0

    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    vis.clear_geometries()
    vis.add_geometry(pcd)

    vis.poll_events()
    vis.update_renderer()

    # -------------------------
    # Display windows
    # -------------------------
    annotated = results[0].plot()

    cv2.imshow("ArchEV YOLO Detection", annotated)
    cv2.imshow("ArchEV Depth Map", depth_vis)

    if cv2.waitKey(1)==27:
        break

cap.release()
vis.destroy_window()
cv2.destroyAllWindows()
