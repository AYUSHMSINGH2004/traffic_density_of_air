import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO

# --- FILE SELECTION UI ---
root = tk.Tk()
root.withdraw()

print("Waiting for video selection...")
video_path = filedialog.askopenfilename(
    title="Select Traffic Video for Analysis",
    filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*")]
)

if not video_path:
    print("No video selected. Exiting program...")
    exit()

print(f"Loading video: {video_path}")
# -------------------------

# 1. Load the pre-trained YOLO model
model = YOLO('yolov8n.pt')

# 2. Define the COCO classes and their CO2 values (g/km)
VEHICLE_DATA = {
    2: {'name': 'Car', 'co2': 120, 'color': (255, 100, 100)},    # Light Blue
    3: {'name': 'Bike', 'co2': 80, 'color': (100, 255, 100)},    # Light Green
    5: {'name': 'Bus', 'co2': 800, 'color': (50, 255, 255)},     # Yellow
    7: {'name': 'Truck', 'co2': 1000, 'color': (50, 50, 255)}    # Red
}

vehicle_counts = {'Car': 0, 'Bike': 0, 'Bus': 0, 'Truck': 0}
total_co2 = {'Car': 0, 'Bike': 0, 'Bus': 0, 'Truck': 0}
tracked_ids = set() 

# 3. Setup the Full-Screen Window
cv2.namedWindow("Air Quality Estimation", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Air Quality Estimation", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# 4. Open the video
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    success, original_frame = cap.read()
    if not success:
        break

    # ==========================================
    # 5. LAYOUT MATH (Based on your sketch)
    # ==========================================
    # We create a solid black 1920x1080 canvas
    canvas = np.zeros((1080, 1920, 3), dtype=np.uint8)
    
    # We dedicate the bottom-right area for the video.
    # Video offsets: 450 pixels from left, 120 pixels from top
    # This leaves the top row and left column blank, as sketched.
    video_width = 1920 - 450  # 1470 pixels wide
    video_height = 1080 - 120 # 960 pixels high
    
    # Resize the video frame to fit this exact area
    frame = cv2.resize(original_frame, (video_width, video_height))

    # 6. Run YOLO tracking on the resized frame
    results = model.track(frame, persist=True, classes=[2, 3, 5, 7], verbose=False)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        class_ids = results[0].boxes.cls.int().cpu().tolist()

        for box, track_id, class_id in zip(boxes, track_ids, class_ids):
            vehicle_info = VEHICLE_DATA[class_id]
            v_name = vehicle_info['name']
            v_co2 = vehicle_info['co2']
            color = vehicle_info['color']

            if track_id not in tracked_ids:
                tracked_ids.add(track_id)
                vehicle_counts[v_name] += 1
                total_co2[v_name] += v_co2

            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(frame, f"{v_name} {track_id}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 7. Paste the processed video frame onto the black canvas
    canvas[120:1080, 450:1920] = frame
    
    # Draw a thin grey border around the video to separate it from the blank areas
    cv2.rectangle(canvas, (450, 120), (1919, 1079), (100, 100, 100), 2)

    # ==========================================
    # 8. COUNTER TABLE UI (Top-Left Corner)
    # ==========================================
    # Draw the boundary for the counter table
    cv2.rectangle(canvas, (20, 20), (430, 400), (255, 255, 0), 2) 

    # Headline
    cv2.putText(canvas, "EMISSION MONITOR", (40, 70), 
                cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 0), 2)
    cv2.line(canvas, (40, 90), (410, 90), (200, 200, 200), 1)

    y_offset = 140 
    overall_co2 = sum(total_co2.values())
    
    # Data Rows
    for v_name in vehicle_counts.keys():
        display_name = 'Buses' if v_name == 'Bus' else f"{v_name}s"
        text = f"{display_name}: {vehicle_counts[v_name]}"
        co2_text = f"CO2: {total_co2[v_name]} g/km"
        
        cv2.putText(canvas, text, (40, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 240, 240), 2)
        cv2.putText(canvas, co2_text, (180, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 240, 240), 2)
        y_offset += 45
    
    cv2.line(canvas, (40, y_offset - 10), (410, y_offset - 10), (200, 200, 200), 1)
    
    # Total Score
    cv2.putText(canvas, "TOTAL OVERALL CO2:", (40, y_offset + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(canvas, f"{overall_co2} g/km", (40, y_offset + 70), 
                cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 3)

    # 9. Show the finalized canvas
    cv2.imshow("Air Quality Estimation", canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# 10. PRINT FINAL SUMMARY TABLE TO TERMINAL
print("\n" + "="*55)
print(f"{'FINAL TRAFFIC & EMISSION REPORT':^55}")
print("="*55)
print(f"{'Vehicle Type':<15} | {'Total Count':<15} | {'Total CO2 (g/km)':<15}")
print("-" * 55)

for v_name in vehicle_counts.keys():
    display_name = 'Buses' if v_name == 'Bus' else f"{v_name}s"
    count = vehicle_counts[v_name]
    co2 = total_co2[v_name]
    print(f"{display_name:<15} | {count:<15} | {co2:<15}")

print("-" * 55)
total_vehicles = sum(vehicle_counts.values())
final_overall_co2 = sum(total_co2.values())
print(f"{'OVERALL TOTAL':<15} | {total_vehicles:<15} | {final_overall_co2:<15}")
print("="*55 + "\n")