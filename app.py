import gradio as gr
import cv2
import numpy as np
from ultralytics import YOLO


model = YOLO("yolov8n.pt")


VEHICLE_DATA = {
    2: {"name": "Car", "co2": 120},
    3: {"name": "Bike", "co2": 80},
    5: {"name": "Bus", "co2": 800},
    7: {"name": "Truck", "co2": 1000}
}


def process_video(video):

    vehicle_counts = {
        "Car":0,
        "Bike":0,
        "Bus":0,
        "Truck":0
    }

    total_co2 = {
        "Car":0,
        "Bike":0,
        "Bus":0,
        "Truck":0
    }


    cap = cv2.VideoCapture(video)

    output = []

    while cap.isOpened():

        success, frame = cap.read()

        if not success:
            break


        results = model.track(
            frame,
            persist=True,
            classes=[2,3,5,7],
            verbose=False
        )


        if results[0].boxes.id is not None:

            boxes = results[0].boxes.xyxy.cpu()
            ids = results[0].boxes.id.int().cpu()
            classes = results[0].boxes.cls.int().cpu()


            for box,track_id,class_id in zip(boxes,ids,classes):

                info = VEHICLE_DATA[int(class_id)]

                name = info["name"]

                vehicle_counts[name]+=1
                total_co2[name]+=info["co2"]


                x1,y1,x2,y2 = map(int,box)

                cv2.rectangle(
                    frame,
                    (x1,y1),
                    (x2,y2),
                    (0,255,0),
                    2
                )


                cv2.putText(
                    frame,
                    name,
                    (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    .8,
                    (0,255,0),
                    2
                )


        output.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))


    cap.release()


    report = ""

    total = 0

    for k in vehicle_counts:

        report += (
            f"{k}: {vehicle_counts[k]} "
            f"CO2={total_co2[k]} g/km\n"
        )

        total += total_co2[k]


    report += f"\nTOTAL CO2: {total} g/km"


    return output, report



demo = gr.Interface(

    fn=process_video,

    inputs=gr.Video(
        label="Upload Traffic Video"
    ),

    outputs=[
        gr.Gallery(
            label="Processed Frames"
        ),
        gr.Textbox(
            label="Emission Report"
        )
    ],

    title="🚦 Traffic CO2 Emission Monitor",
    description=
    "YOLOv8 vehicle detection and CO2 estimation"

)


demo.launch()
