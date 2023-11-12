# Import necessary libraries (e.g., OpenCV, YOLOv5)
import cv2
import torch
# Add any other relevant imports

# Define a label mapping from class IDs to real names
label_mapping = {
    0: "Player 1",
    1: "Player 2",
    # Add mappings for other players as needed
}

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Initialize the video capture or read video frames from a file
video_capture = cv2.VideoCapture("your_video.mp4")

# Main loop for object detection and visualization
while True:
    # Read a frame from the video
    ret, frame = video_capture.read()

    if not ret:
        break  # Break the loop when the video ends

    # Perform object detection on the frame
    results = model(frame)

    # Initialize frame_image for visualization
    frame_image = frame.copy()

    # Iterate through the detection results
    for detections in results.xyxy:
        # Label assignment and visualization
        for det in detections:
            x1, y1, x2, y2, conf, class_id = det
            label = label_mapping.get(class_id.item(), "Unknown")
            frame_image = cv2.rectangle(frame_image, (int(x1), int(y1)), (int(x2), int(y2), (0, 255, 0), 2)
            frame_image = cv2.putText(frame_image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with detection results
    cv2.imshow('Video with Detections', frame_image)

    # Check for user input to exit the loop (e.g., press 'q' to quit)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
