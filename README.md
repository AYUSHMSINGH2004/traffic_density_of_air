# 🚦 Traffic CO₂ Emission Monitor

An AI-powered computer vision system that detects, tracks, and classifies vehicles from traffic videos to estimate **real-time carbon dioxide (CO₂) emissions** using **YOLOv8** and **OpenCV**.

---

## 🌐 Live Demo

🚀 Try the deployed web application here:

👉 https://ayushmsingh2004-traffic-co2-monitor.hf.space

---

## 📌 Project Overview

Traffic congestion is one of the major contributors to urban air pollution and greenhouse gas emissions. This project leverages **Artificial Intelligence**, **Computer Vision**, and **Deep Learning** to monitor traffic density and estimate environmental impact in real time.

The system analyzes uploaded traffic videos, detects different vehicle types, and calculates estimated CO₂ emissions based on vehicle-specific emission metrics.

This enables:
- Smart traffic monitoring
- Environmental impact analysis
- Carbon footprint estimation
- Sustainable urban planning

---

## ✨ Key Features

### 🚗 Real-Time Vehicle Detection
Detects and classifies vehicles using **YOLOv8 (Ultralytics)**.

### 📊 CO₂ Emission Analytics
Calculates estimated CO₂ emissions dynamically based on vehicle type.

### 🎥 Video Processing Dashboard
Provides:
- AI-annotated output video
- Vehicle detection statistics
- Emission analysis
- Summary report

### 🌐 Cloud Deployment
Fully deployed as an interactive web application using **Hugging Face Spaces**.

---

## 🛠️ Tech Stack

### Machine Learning
- YOLOv8 (Ultralytics)

### Computer Vision
- OpenCV
- NumPy

### Frontend / UI
- Gradio
- Custom HTML / CSS Dashboard

### Deployment
- Hugging Face Spaces

---

## 🚘 Vehicle Emission Metrics

| Vehicle Type | Estimated CO₂ Emission |
|--------------|------------------------|
| 🏍 Motorcycle | 80 g/km |
| 🚗 Car | 120 g/km |
| 🚌 Bus | 800 g/km |
| 🚚 Truck | 1000 g/km |

---

## ⚙️ Working Pipeline

```text
Traffic Video Input
        ↓
YOLOv8 Vehicle Detection
        ↓
Vehicle Classification
        ↓
CO₂ Emission Calculation
        ↓
Video Annotation + Report Generation
```

---

## 📂 Project Structure

```bash
traffic_density_of_air/
│
├── app.py                 # Main deployed Gradio app
├── requirements.txt      # Project dependencies
├── README.md
├── yolov8n.pt            # YOLO model weights
├── Data/                 # Sample traffic videos
└── traffic_co2.py        # Local desktop version
```

---

## ▶️ Installation & Setup (Local)

Clone the repository:

```bash
git clone https://github.com/AYUSHMSINGH2004/traffic_density_of_air.git
cd traffic_density_of_air
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run locally:

```bash
python app.py
```

---

## 🎯 Use Cases

- Smart City Traffic Monitoring
- Pollution Analysis
- Environmental Research
- Urban Planning
- Carbon Footprint Estimation
- AI-Based Sustainability Solutions

---

## 📸 Demo

Use the live application:

👉 https://ayushmsingh2004-traffic-co2-monitor.hf.space

Upload a traffic video and receive:

✅ Processed AI-annotated video  
✅ Vehicle detection statistics  
✅ CO₂ emission analysis report  

Sample traffic videos:

👉 https://drive.google.com/drive/folders/1wLa29FiB51TicFNICPSvkHXmiY3Wn7Y_?usp=sharing

---

## ⚠️ Notes

- Large sample videos are excluded due to GitHub size limitations.
- For best performance, use compressed `.mp4` videos.
- Emission values are approximate and intended for research/demo purposes.

---

## 🚀 Future Improvements

- Live CCTV Feed Integration
- Advanced Real-Time Analytics Dashboard
- More Accurate Emission Modeling
- Vehicle Speed Estimation
- Multi-Camera Monitoring
- Cloud GPU Optimization

---

## 👨‍💻 Contributors

| Contributor | Responsibilities |
|------------|------------------|
| Ayush M Singh | Project Development, AI Integration, Deployment |
| Venkata Sriram Topalli | Project Development, AI Integration, System Design, Testing |

---

## 📄 License

This project is developed for **academic, research, and educational purposes**.

---

## ⭐ Support

If you found this project useful:

- Star the repository ⭐
- Fork the project 🍴
- Share feedback 💡
