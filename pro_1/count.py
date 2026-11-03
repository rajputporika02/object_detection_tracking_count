import cv2
from ultralytics import YOLO
import cvzone
import math 
from sort import *

# Load video file
video_path = r"pro_1\videoplayback.webm" 
cap = cv2.VideoCapture(video_path) 

mask_img=cv2.imread(r'pro_1\Mask.png')

# Set video dimensions
cap.set(3, 640)
cap.set(4, 480)

# Load YOLO model
model = YOLO("../yolo_weights/yolov8n.pt")

classNames = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife",
    "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet",
    "tvmonitor", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]

# Check whether the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit() 

tracker=Sort(max_age=20, min_hits=3, iou_threshold=0.3)  
total_count=0
limits=[183,639,1242,639] 
id_Count_list=[]
while True:
    success, img = cap.read()   
    img_final=cv2.bitwise_and(img,mask_img)
    detections=np.empty((0,5))
    # Stop when the video ends
    if not success:
        print("Video finished.")
        break

    # Run YOLO detection
    results = model(img_final, stream=True, verbose=False,device='cpu')
    
    for r in results: ###frame 1, frame 2....=results
        boxes = r.boxes

        for box in boxes:
            # Bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            w, h = x2 - x1, y2 - y1

        

            # Confidence score
            conf = math.ceil(box.conf[0].item() * 100) / 100

            # Class name
            cls_ = int(box.cls[0].item()) 

            if classNames[cls_]=='car' and conf>0.3: 
                # Draw corner bounding box
                '''cvzone.cornerRect(
                img_final, (x1, y1, w, h),
                l=9, rt=5,
                )
                # Display class name and confidence'''
                '''
                cvzone.putTextRect(
                img_final,
                f'{classNames[cls_]} {conf}',
                (max(0, x1), max(35, y1)),
                scale=0.6,
                thickness=1,
                offset=3
                ) 
                '''
                detections=np.vstack((detections,np.array([x1,y1,x2,y2,conf])))
          
    
        tracker_final=tracker.update(detections)   
        cv2.line(img_final,(limits[0],limits[1]),(limits[2],limits[3]),(0,0,255),5)
        for result in tracker_final:
            x1,y1,x2,y2,id=result  
            #print(result)   
            x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
            w,h=x2-x1,y2-y1 
            cvzone.cornerRect(
                img_final, (x1, y1, w, h),
                l=9, rt=2,colorR=(255,0,0)
                ) 
            cvzone.putTextRect(
                img_final,
                f'{int(id)}',
                (max(0, x1), max(35, y1)),
                scale=0.6,
                thickness=1,
                offset=3,colorR=(255,0,0)
                ) 
            cx,cy=x1+w//2,y1+h//2 
            cv2.circle(img_final,(cx,cy),5,(255,0,0),cv2.FILLED) 
            if limits[0]<cx<limits[2] and limits[1]-15<cy<limits[1]+15:
                total_count=total_count+1  
                id_Count_list.append(int(id)) 
                cv2.line(img_final,(limits[0],limits[1]),(limits[2],limits[3]),(0,255,0),5)
        cvzone.putTextRect(
            img_final,
            f'Total Count: {len(list(set(id_Count_list)))}',
            (50, 50),
            scale=1,
            thickness=2,
            offset=10,colorR=(0,0,255)
            ) 
        

 
    # Display video with live detection
    cv2.imshow("YOLO Video Detection with Mask", img_final)
    if cv2.waitKey(1)& 0xFF == ord('q'): 
        break 
cap.release() 
cv2.destroyAllWindows()  

print('-----------------------------------------------------')
print(list(set(id_Count_list))) 
print('Total Count:',len(list(set(id_Count_list))))