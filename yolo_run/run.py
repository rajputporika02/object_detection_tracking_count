from ultralytics import YOLO
import cv2
model=YOLO("yolo_weights\yolov8l.pt")
results=model('yolo_run\img1.jpg',show=True)
cv2.imshow('YOLO Detection', results[0].plot())
cv2.waitKey(0)
cv2.imwrite('yolo_run\img1_result.jpg', results[0].plot())