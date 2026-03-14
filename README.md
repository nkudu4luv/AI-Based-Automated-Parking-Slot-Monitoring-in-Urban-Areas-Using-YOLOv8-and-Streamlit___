# AI-Based-Automated-Parking-Slot-Monitoring-in-Urban-Areas-Using-YOLOv8-and-Streamlit___

# AI-Based Automated Parking Slot Monitoring System

## Project Overview

This project is an **AI-driven automated parking slot monitoring system** designed for urban environments. It uses **YOLOv8 for real-time vehicle detection** and **Streamlit for an interactive dashboard**. The system monitors parking slot occupancy in real time, provides notifications for newly occupied slots, and captures snapshots for record-keeping.

The layout is optimized for usability: the **camera feed appears at the top-left**, while **control buttons (Start, Stop, Reset) and dashboard metrics** are displayed on the right for easy access.

---

## Features

- **Real-Time Vehicle Detection**: Detects cars, trucks, buses, motorcycles, and bicycles using YOLOv8.
- **Interactive Dashboard**: Displays live parking slot status with colored indicators (green = free, red = occupied).
- **Start / Stop / Reset Controls**: All buttons are positioned at the top-right for quick access.
- **Vehicle Notifications**: Alerts when a parking slot becomes occupied.
- **Snapshots**: Automatically saves images of newly occupied slots with timestamped filenames.
- **Dashboard Metrics**: Shows total slots, available slots, and occupied slots.
- **Optimized Layout**: Camera feed on left, controls and dashboard on right.

---

## Folder Structure

```text
ParkingMonitoring/
│
├── app.py                  # Main Streamlit application
├── parking_spaces.json     # JSON file storing slot occupancy
├── snapshots/              # Folder for saving captured snapshots of vehicles
├── yolov8n.pt              # YOLOv8 model file
├── README.md               # Project documentation
└── requirements.txt        # Python dependencies
