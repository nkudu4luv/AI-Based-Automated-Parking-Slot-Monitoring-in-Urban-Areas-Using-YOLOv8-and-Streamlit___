import streamlit as st
import streamlit.components.v1 as components
import cv2
import json
import os
from ultralytics import YOLO
from datetime import datetime

# -----------------------------
# SETTINGS
# -----------------------------
model = YOLO("yolov8n.pt")
SLOT_FILE = "parking_spaces.json"

# Ensure snapshots folder exists
if not os.path.exists("snapshots"):
    os.makedirs("snapshots")

# Ensure parking slot file exists
if not os.path.exists(SLOT_FILE):
    default_slots = {"slot1": False, "slot2": False, "slot3": False, "slot4": False}
    with open(SLOT_FILE, "w") as f:
        json.dump(default_slots, f)

# -----------------------------
# LOAD & SAVE SLOTS
# -----------------------------
def load_slots():
    with open(SLOT_FILE, "r") as f:
        return json.load(f)

def save_slots(data):
    with open(SLOT_FILE, "w") as f:
        json.dump(data, f, indent=4)

# -----------------------------
# STREAMLIT SETTINGS
# -----------------------------
st.set_page_config(page_title="Smart Parking", layout="wide")
st.markdown("""
<style>
.stApp{background: linear-gradient(120deg,#89f7fe,#66a6ff);}
h1{text-align:center;color:white;}
.button-row button{
    padding:10px 20px; margin-bottom:10px; font-size:16px; border:none; border-radius:8px; cursor:pointer;
    background-color:#333; color:white; transition:0.3s; width:100%;
}
.button-row button:hover{background-color:#555;}
</style>
""", unsafe_allow_html=True)

st.title("🚗 Smart Parking Monitoring System")

# -----------------------------
# SESSION STATE
# -----------------------------
if "run_camera" not in st.session_state:
    st.session_state.run_camera = False

# -----------------------------
# PLACEHOLDERS
# -----------------------------
vehicle_indicator = st.empty()
notification_placeholder = st.empty()

# -----------------------------
# TOP LAYOUT: CAMERA LEFT | CONTROL + DASHBOARD RIGHT
# -----------------------------
col_camera, col_right = st.columns([2,3])
frame_placeholder = col_camera.empty()

with col_right:
    # Buttons
    st.markdown('<div class="button-row">', unsafe_allow_html=True)
    start_clicked = st.button("Start Monitoring")
    stop_clicked = st.button("Stop Monitoring")
    reset_clicked = st.button("Reset Parking System")
    st.markdown('</div>', unsafe_allow_html=True)

    # Dashboard metrics
    slots = load_slots()
    available = sum(1 for s in slots.values() if not s)
    occupied = sum(1 for s in slots.values() if s)

    mcol1, mcol2, mcol3 = st.columns(3)
    mcol1.metric("Total Slots", len(slots))
    mcol2.metric("Available", available)
    mcol3.metric("Occupied", occupied)

    # Parking Status
    st.subheader("Parking Status")
    grid1, grid2 = st.columns(2)
    for i,(slot,status) in enumerate(slots.items()):
        col = grid1 if i%2==0 else grid2
        if status:
            col.error(f"{slot} 🔴 Occupied")
        else:
            col.success(f"{slot} 🟢 Available")

# -----------------------------
# DETECTION FUNCTION WITH SNAPSHOTS
# -----------------------------
def run_detection():
    cap = cv2.VideoCapture(0)
    parking_areas = {
        "slot1": (50,100,250,300),
        "slot2": (300,100,500,300),
        "slot3": (50,350,250,550),
        "slot4": (300,350,500,550)
    }
    prev_status = load_slots()

    while st.session_state.run_camera:
        slots = load_slots()
        ret, frame = cap.read()
        if not ret:
            st.error("Camera not detected")
            break

        results = model(frame)
        detected_objects = []
        vehicle_detected = False

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]
                if label in ["car","truck","bus","motorbike","bicycle"]:
                    x1,y1,x2,y2 = map(int, box.xyxy[0])
                    detected_objects.append((x1,y1,x2,y2))
                    vehicle_detected = True
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                    cv2.putText(frame,label,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

        # Check parking slots
        for slot_name,(sx1,sy1,sx2,sy2) in parking_areas.items():
            occupied = any(x1 < sx2 and x2 > sx1 and y1 < sy2 and y2 > sy1 for (x1,y1,x2,y2) in detected_objects)
            
            # If slot just became occupied → save snapshot
            if occupied and not prev_status[slot_name]:
                notification_placeholder.error(f"🚨 {slot_name} is now Occupied!")
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                filename = f"snapshots/{slot_name}_{timestamp}.jpg"
                cv2.imwrite(filename, frame)

            slots[slot_name] = occupied
            prev_status[slot_name] = occupied
            color = (0,255,0) if not occupied else (0,0,255)
            cv2.rectangle(frame,(sx1,sy1),(sx2,sy2),color,3)
            cv2.putText(frame,slot_name,(sx1,sy1-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)

        save_slots(slots)
        if vehicle_detected:
            vehicle_indicator.success("🚨 Vehicle Detected!")
        else:
            vehicle_indicator.info("No vehicle detected 🟢")

        frame_placeholder.image(frame, channels="BGR")

    cap.release()

# -----------------------------
# HANDLE BUTTON CLICKS
# -----------------------------
if start_clicked:
    st.session_state.run_camera = True
    run_detection()

if stop_clicked:
    st.session_state.run_camera = False

if reset_clicked:
    slots = load_slots()
    reset_data = {k:False for k in slots}
    save_slots(reset_data)
    st.success("Parking system reset!")