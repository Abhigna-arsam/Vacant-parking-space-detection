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
    [(217,352),(219,422),(273,418),(261,347)],
    [(274,345),(286,417),(338,415),(321,345)],
    [(336,343),(357,410),(409,408),(382,340)],
    [(396,338),(426,404),(479,399),(439,334)],
    [(458,333),(494,397),(543,390),(495,330)],
    [(511,327),(557,388),(603,383),(549,324)],
    [(564,323),(615,381),(654,372),(596,315)],
    [(616,316),(666,369),(703,363),(642,312)],
    [(674,311),(730,360),(764,355),(707,308)],
    
]



# Create a dictionary to store parking information
parking_info = {i: {"detected": False, "payment": 0} for i in range(len(areas))}

model = YOLO('yolov8s.pt')

def generate_payment(area_number):
    # Generate payment for each detected car (e.g., 1 unit per millisecond)
    #payment = int(time.time() * 1000)  # generate payment in milliseconds
    #parking_info[area_number]["payment"] += payment
    #print(f"Area {area_number+1}: Payment generated - {payment} units")
    payment_per_second = 1  # Define the payment per second
    current_time = int(time.time())  # Get the current time in seconds
    last_payment_time = parking_info[area_number].get("last_payment_time", current_time)
    elapsed_time = current_time - last_payment_time  # Calculate the elapsed time since last payment

    payment = payment_per_second * elapsed_time  # Calculate payment based on elapsed time
    parking_info[area_number]["payment"] += payment
    parking_info[area_number]["last_payment_time"] = current_time  # Update last payment time

    print(f"Area {area_number+1}: Payment generated - {payment} units")

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture('parking1.mp4')

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    time.sleep(1)
    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    list1=[]
    list2=[]
    list3=[]
    list4=[]
    list5=[]
    list6=[]
    list7=[]
    list8=[]
    list9=[]
    list10=[]
    list11=[]
    list12=[]

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]

        
        if 'car' in c:
            cx = int(x1 + x2) // 2
            cy = int(y1 + y2) // 2

           # for area_number, area in enumerate(areas):
            area_number=0
            results1 = cv2.pointPolygonTest(np.array(areas[0], np.int32), ((cx, cy)), False)
            if results1 >= 0:
                parking_info[area_number]["detected"] = True
                generate_payment(area_number)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                list1.append(c)
                cv2.putText(frame, str(c), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
            area_number=1
            results2=cv2.pointPolygonTest(np.array(areas[1],np.int32),((cx,cy)),False)
            if results2>=0:
                parking_info[area_number]["detected"] = True
                generate_payment(area_number)
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
                list2.append(c)
            area_number=2
            results3=cv2.pointPolygonTest(np.array(areas[2],np.int32),((cx,cy)),False)
            if results3>=0:
                parking_info[area_number]["detected"] = True
                generate_payment(area_number)
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
                list3.append(c)
            area_number=3
            results4=cv2.pointPolygonTest(np.array(areas[3],np.int32),((cx,cy)),False)
            if results4>=0:
                parking_info[area_number]["detected"] = True
                generate_payment(area_number)
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
                list4.append(c)
            area_number=4
            results5=cv2.pointPolygonTest(np.array(areas[4],np.int32),((cx,cy)),False)
            if results5>=0:
                parking_info[area_number]["detected"] = True
                generate_payment(area_number)
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
                list5.append(c)
            area_number=5
            results6=cv2.pointPolygonTest(np.array(areas[5],np.int32),((cx,cy)),False)
            if results6>=0:
                parking_info[area_number]["detected"] = True
                generate_payment(area_number)
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
                list6.append(c)
            area_number=6
            results7=cv2.pointPolygonTest(np.array(areas[6],np.int32),((cx,cy)),False)
            if results7>=0:
                parking_info[area_number]["detected"] = True
                generate_payment(area_number)
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
                list7.append(c)
            area_number=7
            results8=cv2.pointPolygonTest(np.array(areas[7],np.int32),((cx,cy)),False)
            if results8>=0:
                parking_info[area_number]["detected"] = True
                generate_payment(area_number)
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
                list8.append(c)
            area_number=8
            results9=cv2.pointPolygonTest(np.array(areas[8],np.int32),((cx,cy)),False)
            if results9>=0:
                parking_info[area_number]["detected"] = True
                generate_payment(area_number)
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
                list9.append(c)
            area_number=9
            results10=cv2.pointPolygonTest(np.array(areas[9],np.int32),((cx,cy)),False)
            if results10>=0:
                parking_info[area_number]["detected"] = True
                generate_payment(area_number)
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
                list10.append(c)
            area_number=10
            results11=cv2.pointPolygonTest(np.array(areas[10],np.int32),((cx,cy)),False)
            if results11>=0:
                parking_info[area_number]["detected"] = True
                generate_payment(area_number)
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
                list11.append(c)
            area_number=11
            results12=cv2.pointPolygonTest(np.array(areas[11],np.int32),((cx,cy)),False)
            if results12>=0:
                parking_info[area_number]["detected"] = True
                generate_payment(area_number)
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
                list12.append(c)
            
    a1=(len(list1))
    a2=(len(list2))       
    a3=(len(list3))    
    a4=(len(list4))
    a5=(len(list5))
    a6=(len(list6)) 
    a7=(len(list7))
    a8=(len(list8)) 
    a9=(len(list9))
    a10=(len(list10))
    a11=(len(list11))
    a12=(len(list12))
    o=(a1+a2+a3+a4+a5+a6+a7+a8+a9+a10+a11+a12)
    space=(12-o)
    print("available slots: "+str(space))
    #audio_file = "nice_message_alert.mp3"
    if space < 8:
        frequency=3000
        duration=6000
        winsound.Beep(frequency,duration)

    
    
    if a1==1:
        cv2.polylines(frame,[np.array(areas[0],np.int32)],True,(0,0,255),2)
        cv2.putText(frame,str('1'),(50,441),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
    else:
        cv2.polylines(frame,[np.array(areas[0],np.int32)],True,(0,255,0),2)
        cv2.putText(frame,str('1'),(50,441),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
    if a2==1:
        cv2.polylines(frame,[np.array(areas[1],np.int32)],True,(0,0,255),2)
        cv2.putText(frame,str('2'),(106,440),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
    else:
        cv2.polylines(frame,[np.array(areas[1],np.int32)],True,(0,255,0),2)
        cv2.putText(frame,str('2'),(106,440),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
    if a3==1:
        cv2.polylines(frame,[np.array(areas[2],np.int32)],True,(0,0,255),2)
        cv2.putText(frame,str('3'),(175,436),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
    else:
        cv2.polylines(frame,[np.array(areas[2],np.int32)],True,(0,255,0),2)
        cv2.putText(frame,str('3'),(175,436),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
    if a4==1:
        cv2.polylines(frame,[np.array(areas[3],np.int32)],True,(0,0,255),2)
        cv2.putText(frame,str('4'),(250,436),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
    else:
        cv2.polylines(frame,[np.array(areas[3],np.int32)],True,(0,255,0),2)
        cv2.putText(frame,str('4'),(250,436),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
    if a5==1:
        cv2.polylines(frame,[np.array(areas[4],np.int32)],True,(0,0,255),2)
        cv2.putText(frame,str('5'),(315,429),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
    else:
        cv2.polylines(frame,[np.array(areas[4],np.int32)],True,(0,255,0),2)
        cv2.putText(frame,str('5'),(315,429),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
    if a6==1:
        cv2.polylines(frame,[np.array(areas[5],np.int32)],True,(0,0,255),2)
        cv2.putText(frame,str('6'),(386,421),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
    else:
        cv2.polylines(frame,[np.array(areas[5],np.int32)],True,(0,255,0),2)
        cv2.putText(frame,str('6'),(386,421),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1) 
    if a7==1:
        cv2.polylines(frame,[np.array(areas[6],np.int32)],True,(0,0,255),2)
        cv2.putText(frame,str('7'),(456,414),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
    else:
        cv2.polylines(frame,[np.array(areas[6],np.int32)],True,(0,255,0),2)
        cv2.putText(frame,str('7'),(456,414),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
    if a8==1:
        cv2.polylines(frame,[np.array(areas[7],np.int32)],True,(0,0,255),2)
        cv2.putText(frame,str('8'),(527,406),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
    else:
        cv2.polylines(frame,[np.array(areas[7],np.int32)],True,(0,255,0),2)
        cv2.putText(frame,str('8'),(527,406),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)  
    if a9==1:
        cv2.polylines(frame,[np.array(areas[8],np.int32)],True,(0,0,255),2)
        cv2.putText(frame,str('9'),(591,398),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
    else:
        cv2.polylines(frame,[np.array(areas[8],np.int32)],True,(0,255,0),2)
        cv2.putText(frame,str('9'),(591,398),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
    if a10==1:
        cv2.polylines(frame,[np.array(areas[9],np.int32)],True,(0,0,255),2)
        cv2.putText(frame,str('10'),(649,384),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
    else:
        cv2.polylines(frame,[np.array(areas[9],np.int32)],True,(0,255,0),2)
        cv2.putText(frame,str('10'),(649,384),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
    if a11==1:
        cv2.polylines(frame,[np.array(areas[10],np.int32)],True,(0,0,255),2)
        cv2.putText(frame,str('11'),(697,377),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
    else:
        cv2.polylines(frame,[np.array(areas[10],np.int32)],True,(0,255,0),2)
        cv2.putText(frame,str('11'),(697,377),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
    if a12==1:
        cv2.polylines(frame,[np.array(areas[11],np.int32)],True,(0,0,255),2)
        cv2.putText(frame,str('12'),(752,371),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
    else:
        cv2.polylines(frame,[np.array(areas[11],np.int32)],True,(0,255,0),2)
        cv2.putText(frame,str('12'),(752,371),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)

   
    
    cv2.putText(frame,"Available slots: "+str(space),(23,30),cv2.FONT_HERSHEY_PLAIN,3,(255,255,255),2)

    cv2.imshow("RGB", frame)
    # Display payment information
    for area_number, info in parking_info.items():
        if info["detected"]:
            print(f"Area {area_number+1}: Payment - {info['payment']} units")

    cv2.imshow("RGB", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
