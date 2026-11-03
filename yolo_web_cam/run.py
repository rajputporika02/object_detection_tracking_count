#######on external video
import cv2
from ultralytics import YOLO
import cvzone
import math

# Load video file
video_path = r"yolo_web_cam\videoplayback.webm"
cap = cv2.VideoCapture(video_path)

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

while True:
    success, img = cap.read()

    # Stop when the video ends
    if not success:
        print("Video finished.")
        break

    # Run YOLO detection
    results = model(img, stream=True, verbose=False)

    for r in results:
        boxes = r.boxes

        for box in boxes:
            # Bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            w, h = x2 - x1, y2 - y1

            # Draw corner bounding box
            cvzone.cornerRect(
                img, (x1, y1, w, h),
                l=9, t=3, rt=1,
                colorR=(255, 0, 255)
            )

            # Confidence score
            conf = math.ceil(box.conf[0].item() * 100) / 100

            # Class name
            cls_ = int(box.cls[0].item())

            # Display class name and confidence
            cvzone.putTextRect(
                img,
                f'{classNames[cls_]} {conf}',
                (max(0, x1), max(35, y1)),
                scale=1,
                thickness=1,
                offset=3
            )

    # Display video with live detection
    cv2.imshow("YOLO Video Detection", img)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video and close windows
cap.release()
cv2.destroyAllWindows()