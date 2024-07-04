import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import time
import winsound

# Define areas as a list of coordinates 
areas = [
    [(52, 364), (30, 417), (73, 412), (88, 369)],
    [(105, 353), (86, 428), (137, 427), (146, 358)],
    [(159, 354), (150, 427), (204, 425), (203, 353)],
    [(217, 352), (219, 422), (273, 418), (261, 347)],
    [(274, 345), (286, 417), (338, 415), (321, 345)],
    [(336, 343), (357, 410), (409, 408), (382, 340)],
    [(396, 338), (426, 404), (479, 399), (439, 334)],
    [(458, 333), (494, 397), (543, 390), (495, 330)],
    [(511, 327), (557, 388), (603, 383), (549, 324)],
    [(564, 323), (615, 381), (654, 372), (596, 315)],
    [(616, 316), (666, 369), (703, 363), (642, 312)],
    [(674, 311), (730, 360), (764, 355), (707, 308)],
]

# Create a dictionary to store parking information
parking_info = {i: {"detected": False, "entry_time": None, "exit_time": None, "payment": 0} for i in range(len(areas))}

#Loading a pre-trained YOLOv8 model for detecting objects 
model = YOLO('yolov8s.pt')

#Payment Calculation Function
def generate_payment(area_number):
    payment_per_second = 1  # Define the payment per second
    entry_time = parking_info[area_number]["entry_time"]
    exit_time = parking_info[area_number]["exit_time"]

    if entry_time and exit_time:
        elapsed_time = exit_time - entry_time  # Calculate the elapsed time in seconds
        payment = payment_per_second * elapsed_time  # Calculate payment based on elapsed time
        parking_info[area_number]["payment"] += payment
        print(f"Area {area_number + 1}: Payment generated - {payment} units")

#Mouse Callback Function
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)

cv2.namedWindow('RGB')
# Sets RGB function as the callback for mouse events in the 'RGB' window
cv2.setMouseCallback('RGB', RGB)

#Video Capture Initialization
cap = cv2.VideoCapture('parking1.mp4')

#Loading Class Labels
my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

#Main Loop for Processing Video Frames
while True:
    ret, frame = cap.read()
    if not ret:
        break
    time.sleep(1)
    frame = cv2.resize(frame, (1020, 500))

    #Object Detection on Frame
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")


    #Analyzing Detected Objects
    detected_areas = set()

    for index, row in px.iterrows():
        x1 = int(row[0])#x-coordinate top-left corner
        y1 = int(row[1])#y-coordinate top-left corner
        x2 = int(row[2])#x-coordinate bottom-right corner
        y2 = int(row[3])# y-coordinate bottom-right corner
        d = int(row[5])
        c = class_list[d]

        if 'car' in c:
            cx = int(x1 + x2) // 2
            cy = int(y1 + y2) // 2

            for area_number, area in enumerate(areas):
                result = cv2.pointPolygonTest(np.array(area, np.int32), (cx, cy), False)
                if result >= 0:
                    #Checks if this parking area (area_number) has not been previously detected (detected is False).
                    if not parking_info[area_number]["detected"]:
                        parking_info[area_number]["entry_time"] = int(time.time())
                    parking_info[area_number]["detected"] = True
                    detected_areas.add(area_number)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                    cv2.putText(frame, str(c), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                    break

    #Updating Parking Information
    for area_number in range(len(areas)):
        if area_number not in detected_areas:
            #If the parking area was previously detected 
            if parking_info[area_number]["detected"]:   
                parking_info[area_number]["exit_time"] = int(time.time())
                generate_payment(area_number)
            parking_info[area_number]["detected"] = False
            parking_info[area_number]["payment"] = 0
            parking_info[area_number]["entry_time"] = None
            parking_info[area_number]["exit_time"] = None

    #Counting Available Slots
    occupied_slots = sum(info["detected"] for info in parking_info.values())
    available_slots = len(areas) - occupied_slots
    print("Available slots: " + str(available_slots))

    #Sound Alert for Limited Parking
    if available_slots < 8:
        frequency = 3000
        duration = 6000
        winsound.Beep(frequency, duration)

    #Drawing Polygons and Text on Frame
    for area_number, area in enumerate(areas):
        if parking_info[area_number]["detected"]:
            cv2.polylines(frame, [np.array(area, np.int32)], True, (0, 0, 255), 2)
        else:
            cv2.polylines(frame, [np.array(area, np.int32)], True, (0, 255, 0), 2)
        cv2.putText(frame, str(area_number + 1), tuple(area[0]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

    cv2.putText(frame, "Available slots: " + str(available_slots), (23, 30), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)

    #Displaying Frame and Exiting
    cv2.imshow("RGB", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
