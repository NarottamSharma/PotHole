import os
import cv2
from ultralytics import YOLO

# Set the DISPLAY environment variable
os.environ['DISPLAY'] = ':99'

# Load the trained YOLO model
try:
    model = YOLO('/workspaces/PotHole/pothole-detection-yolo11/models/best.pt')
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Open the video file
cap = cv2.VideoCapture('/workspaces/PotHole/pothole-detection-yolo11/data/external_data/video_2025-03-18_17-21-26.mp4')

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

output_folder = 'processed_frames'
os.makedirs(output_folder, exist_ok=True)

frame_count = 0
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or failed to capture frame.")
        break

    # Run detection on the frame
    try:
        results = model.predict(frame, imgsz=640, conf=0.5)
    except Exception as e:
        print(f"Error during prediction: {e}")
        continue

    # Process and annotate the frame
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0]
            label = r.names[int(box.cls[0])]
            color = (0, 255, 0)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Save the processed frame as an image
    frame_filename = os.path.join(output_folder, f'frame_{frame_count:04d}.png')
    cv2.imwrite(frame_filename, frame)
    frame_count += 1

    # Display the frame (if running locally with a display)
    cv2.imshow("Pothole Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
