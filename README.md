# 🔌 ArchEV – Autonomous EV Charging System

ArchEV is an intelligent robotic arm system designed to automate Electric Vehicle (EV) charging using computer vision and depth sensing.

## 🚀 Features

* YOLO-based EV charging port detection
* Depth estimation using Intel RealSense
* 3D localization of charging socket
* Autonomous alignment for robotic arm

## 🛠️ Tech Stack

* Python
* OpenCV
* Intel RealSense
* YOLOv8

## 📂 Project Structure

```
src/        → main scripts  
models/     → trained weights  
data/       → dataset/configs  
```

## 📦 Model Weights

Due to size limitations, the trained model is not included.

Download here:
https://drive.google.com/file/d/1OA3mKlKrSxeFnvDRQgOREwrGv_rUF7CC/view?usp=sharing

Place it in:

```
models/best.pt
```

## ▶️ Usage

Run:

```
python src/archev_realsense_yolo_depth.py
```

## 🎯 Objective

To develop an automated EV charging solution that improves convenience and supports smart electric mobility.
