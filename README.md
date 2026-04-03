# 🔌 ArchEV – Autonomous EV Charging System

ArchEV is an intelligent robotic arm system designed to automate Electric Vehicle (EV) charging using computer vision, depth sensing, and custom hardware.

---

## 🚀 Features

* 🔍 YOLO-based EV charging port detection
* 📏 Depth estimation using Intel RealSense
* 🤖 Autonomous robotic alignment for charging
* 🔌 Custom-designed PCB (Altium) for system control

---

## 🛠️ Tech Stack

* Python
* OpenCV
* YOLOv8
* Intel RealSense
* Altium Designer

---

## 📂 Project Structure

```
src/         → Computer vision & control scripts  
hardware/    → PCB design (schematic + layout + BOM)  
models/      → Model weights (external)  
data/        → Dataset/configs  
```

---

## 🔧 Hardware Design

Custom PCB designed in Altium for motor control and system interfacing.

👉 [View Hardware Design](hardware/)
PCB FILES : https://drive.google.com/drive/folders/1RLfGGfEXFfltMMTO_4W8Be5pAfSRfIsN?usp=sharing

---

## 📦 Model Weights

Due to size limitations, the trained model is not included.

👉 Download: (https://drive.google.com/file/d/1OA3mKlKrSxeFnvDRQgOREwrGv_rUF7CC/view?usp=drive_link)

Place in:

```
models/best.pt
```

---

## ▶️ Usage

Run:

```
python src/archev_realsense_yolo_depth.py
```

---

## 🎯 Objective

To build a smart, automated EV charging system that improves user convenience and supports the future of electric mobility.
