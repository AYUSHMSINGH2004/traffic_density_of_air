# 🚦 Traffic CO2 Emission Monitor

An AI-powered computer vision system that monitors traffic and estimates real-time CO2 emissions using YOLOv8. The system detects, tracks, and classifies vehicles from video input and provides a live dashboard of environmental impact.

---

## 📌 Project Overview

This project leverages computer vision to analyze traffic flow and calculate carbon dioxide (CO2) emissions in real time. By detecting different vehicle types and assigning emission values, it provides immediate insights into environmental impact. :contentReference[oaicite:0]{index=0}

---

## ✨ Features

- 🚗 **Real-Time Vehicle Detection & Tracking**  
  Detects and tracks vehicles using YOLOv8.

- 📊 **Dynamic CO2 Emission Calculation**  
  Calculates emissions (g/km) based on vehicle type.

- 🖥️ **Live Dashboard Visualization**  
  Displays bounding boxes, vehicle counts, and total emissions.

- 📄 **Automated Summary Report**  
  Generates final emission statistics after execution.

---

## 🛠️ Tech Stack

- **Machine Learning:** YOLOv8 (Ultralytics)
- **Computer Vision:** OpenCV, NumPy
- **UI Handling:** OpenCV-based custom dashboard
- **File Selection:** Tkinter

---

## 🚘 Vehicle Emission Metrics

| Vehicle | CO2 Emission (g/km) |
|--------|---------------------|
| Bike   | 80                  |
| Car    | 120                 |
| Bus    | 800                 |
| Truck  | 1000                |

---

## ⚙️ How It Works

1. Select a video file (.mp4/.avi)
2. Model detects and tracks vehicles
3. CO2 emissions are calculated in real-time
4. Live dashboard displays results
5. Final summary is printed in terminal :contentReference[oaicite:1]{index=1}

---

## ▶️ Installation & Setup

bash
git clone https://github.com/AYUSHMSINGH2004/traffic_density_of_air.git
cd traffic_density_of_air
pip install -r requirements.txt


▶️ Run the Project

python traffic_co2.py

📂 Project Structure

traffic_density_of_air/
│── Data/                
│── traffic_co2.py       # Main script
│── yolov8n.pt           # Model file
│── README.md
│── .gitignore

📸 Demo

👉 Add your demo video link here (Google Drive / YouTube)

⚠️ Notes

Large video files are not included due to GitHub size limits

Download sample videos from the provided link : https://drive.google.com/drive/folders/1wLa29FiB51TicFNICPSvkHXmiY3Wn7Y_?usp=drive_link

🚀 Future Improvements

1.Web-based dashboard (React + FastAPI)

2.Live CCTV integration

3.More accurate emission modeling

4.Cloud deployment

👨‍💻 Contributors

Ayush M Singh
Venkata Sriram Topalli
Bishal Kumar Mandal
Kanhaiya
Ojas Arora


📄 License

This project is for academic and research purposes.

## 🔥 Why this README is strong
- Looks **professional**
- Easy for **recruiters to understand**
- Highlights **AI + real-world impact**
- Clean structure (very important)

