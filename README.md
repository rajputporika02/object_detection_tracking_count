# YOLO-Based Vehicle Detection, Tracking & Counting

## Project Overview

This project implements a real-time vehicle detection, tracking, and counting system using **YOLOv8, OpenCV, and the SORT tracking algorithm**. The system processes traffic video footage, detects cars, assigns unique tracking IDs, and counts vehicles crossing a predefined counting line.

The project combines deep learning-based object detection with multi-object tracking to maintain vehicle identities across consecutive video frames and avoid duplicate counting.

## Key Features

- **Vehicle Detection:** Uses the pretrained YOLOv8n model to detect vehicles in video frames.
- **Class Filtering:** Identifies and processes only cars with a confidence score greater than 0.30.
- **Object Tracking:** Integrates the SORT algorithm to assign unique IDs to detected vehicles.
- **Region of Interest (ROI):** Applies a binary mask to restrict detection to a selected traffic region.
- **Vehicle Counting:** Counts unique vehicles crossing a predefined virtual counting line.
- **Duplicate Prevention:** Maintains a list of tracked vehicle IDs to prevent repeated counting.
- **Visual Annotation:** Displays bounding boxes, tracking IDs, vehicle positions, and the cumulative vehicle count.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| YOLOv8n | Deep learning-based vehicle detection |
| OpenCV | Video processing, masking, and visualization |
| SORT | Multi-object tracking and ID assignment |
| cvzone | Bounding boxes and text annotations |
| NumPy | Detection array processing |
| PyTorch | Deep learning inference framework |

## Project Architecture

The system follows the pipeline below:

1. **Video Input:** Reads traffic video footage using OpenCV.
2. **ROI Masking:** Applies a predefined mask to focus on the required traffic region.
3. **Object Detection:** YOLOv8n processes each video frame and detects objects.
4. **Class and Confidence Filtering:** Retains car detections with confidence greater than 0.30.
5. **Object Tracking:** SORT associates detections across frames and assigns unique tracking IDs.
6. **Vehicle Position Calculation:** Calculates the centre point of each tracked vehicle using its bounding box coordinates.
7. **Line Crossing Detection:** Checks whether the vehicle centre enters the predefined counting-line region.
8. **Unique Vehicle Counting:** Stores vehicle IDs and eliminates duplicates to calculate the total count.
9. **Visualization:** Displays annotated video frames with tracking IDs and the cumulative vehicle count.

## System Workflow

```text
             Traffic Video
                   |
                   v
          OpenCV Video Capture
                   |
                   v
             ROI Masking
                   |
                   v
          YOLOv8n Detection
                   |
                   v
       Car Class & Confidence Filter
             (Confidence > 0.30)
                   |
                   v
            SORT Tracking
                   |
                   v
          Unique ID Assignment
                   |
                   v
        Vehicle Centre Calculation
                   |
                   v
         Virtual Line Crossing
                   |
                   v
        Duplicate ID Elimination
                   |
                   v
          Total Vehicle Count
                   |
                   v
       Annotated Video Output
```

## Project Structure

```text
YOLO-Based-Vehicle-Detection-Tracking-Counting/
│
├── pro_1/
│   ├── count.py
│   ├── sort.py
│   ├── videoplayback.webm
│   └── Mask.png
│
├── yolo_weights/
│   └── yolov8n.pt
│
├── requirements.txt
└── README.md
```

**File Description:**

- `pro_1/count.py` – Implements vehicle detection, tracking, and counting.
- `pro_1/sort.py` – Contains the SORT tracking algorithm implementation.
- `pro_1/videoplayback.webm` – Input traffic video.
- `pro_1/Mask.png` – Region-of-interest mask.
- `yolo_weights/yolov8n.pt` – Pretrained YOLOv8 nano model.
- `requirements.txt` – Python project dependencies.

*Note: The folder structure above represents the recommended standalone project organization. Adjust filenames and locations to match your actual repository.*

## Installation and Setup

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd YOLO-Based-Vehicle-Detection-Tracking-Counting
```

### 2. Create a Virtual Environment

```bash
python -m venv env
```

Activate the environment:

**Windows PowerShell:**

```powershell
.\env\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
source env/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If the requirements file is unavailable, install the required packages:

```bash
pip install ultralytics opencv-python cvzone numpy scipy filterpy
```

## Model Weights

This project uses the pretrained **YOLOv8n (Nano)** model from Ultralytics.

The model is lightweight and suitable for object detection applications.

Download the pretrained model from:

https://github.com/ultralytics/assets/releases

Place `yolov8n.pt` inside the `yolo_weights/` directory.

## Execution

### 1. Vehicle Detection and Tracking

Run the detection and tracking implementation:

```bash
python pro_1/count.py
```

The system detects cars, assigns tracking IDs, and displays bounding boxes around tracked vehicles.

### 2. Vehicle Counting

The counting implementation uses a predefined virtual line to identify vehicles crossing the counting region.

When a tracked vehicle crosses the counting line:

- Its centre coordinates are evaluated.
- Its tracking ID is added to the counting list.
- Duplicate IDs are eliminated.
- The total number of unique vehicles is updated.

The final count is displayed on the video and printed in the terminal after processing.

**Note:** If detection and tracking are maintained in separate scripts, execute the corresponding script for each functionality.

## GPU and CPU Execution

The original vehicle counting implementation uses CPU-based inference:

```python
results = model(
    img_final,
    stream=True,
    verbose=False,
    device='cpu'
)
```

The YOLO video detection component was also tested separately using Google Colab with a Tesla T4 GPU.

GPU inference can be enabled for the vehicle counting pipeline by changing the device argument to `device=0` in a CUDA-enabled environment.

## Configuration Parameters

| Parameter | Value | Description |
|---|---|---|
| YOLO Model | YOLOv8n | Pretrained object detection model |
| Confidence Threshold | 0.30 | Minimum confidence for car detections |
| SORT max_age | 20 | Maximum frames a track can remain unmatched |
| SORT min_hits | 3 | Minimum detections required for track confirmation |
| SORT IoU Threshold | 0.30 | IoU threshold for associating detections |
| Video Resolution | 640 × 480 | Configured video capture dimensions |
| Detection Class | Car | Target object for vehicle counting |

## Results and Output

The system produces an annotated video displaying:

- Detected vehicles with bounding boxes.
- Unique tracking IDs for individual vehicles.
- Centre points of tracked vehicles.
- A predefined virtual counting line.
- A dynamically updated total vehicle count.

The final output includes the list of unique tracked vehicle IDs and the total number of vehicles counted during the video.

## Applications

- Traffic flow monitoring.
- Vehicle counting at road intersections.
- Intelligent transportation systems.
- Automated traffic surveillance.
- Road traffic analysis.
- Smart city traffic management.

## Future Improvements

- Extend vehicle detection to cars, buses, trucks, and motorcycles.
- Implement bidirectional vehicle counting.
- Add vehicle speed estimation using tracking trajectories.
- Develop real-time traffic monitoring using CCTV cameras.
- Integrate vehicle counting statistics with dashboards.
- Explore GPU acceleration for faster inference.

## Limitations

- Detection accuracy depends on video quality, lighting conditions, and object visibility.
- Occlusion and overlapping vehicles may affect tracking consistency.
- The current counting implementation uses a predefined counting region.
- The current counting pipeline filters only the car class.
- Incorrect tracking associations may affect the final vehicle count.

## References

- **Ultralytics YOLOv8:** https://github.com/ultralytics/ultralytics
- **SORT – Simple Online and Realtime Tracking:** https://github.com/abewley/sort
- **OpenCV:** https://opencv.org/
- **cvzone:** https://github.com/cvzone/cvzone

## Author

**P. Abidsingh Rajput Porika**

Computer Vision | Deep Learning | Machine Learning

## License

This project is intended for educational and research purposes. Refer to the respective licenses of the third-party libraries and pretrained model used in this project.
