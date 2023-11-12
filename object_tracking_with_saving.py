import cv2
import torch
from pathlib import Path
from models.experimental import attempt_load
from utils.general import non_max_suppression
from utils.torch_utils import select_device

# Load YOLOv5 model
weights = 'yolov5s.pt'  # You can replace this with the path to the pre-trained weights
device = select_device('0')
model = attempt_load(weights)
# Set confidence threshold for object detection
conf_threshold = 0.5

# Open the input video (replace 'input_video.mp4' with your video file)
input_video_path = 'fifa.mp4'
cap = cv2.VideoCapture(input_video_path)

# Get video frame dimensions
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the codec and create a VideoWriter object to save the output video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video_path = 'output_video.mp4'
out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = torch.from_numpy(frame).to(device)
    img = img / 255.0  # Normalize the image

    # Perform object detection
    results = model(img.unsqueeze(0))  # Unsqueeze to add a batch dimension
    pred = non_max_suppression(results.pred[0], conf_threshold)

    # Draw bounding boxes for detected objects
    if pred[0] is not None:
        for det in pred[0]:
            x1, y1, x2, y2, conf, cls = det
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    out.write(frame)  # Write the frame to the output video

    cv2.imshow('YOLOv5 Object Tracking', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
out.release()  # Release the output video
cv2.destroyAllWindows()
