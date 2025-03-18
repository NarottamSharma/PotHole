from ultralytics import YOLO
import cv2
import os

def detect_video(video_path, output_path):
    # Check if input video exists
    if not os.path.exists(video_path):
        print(f"Error: Input video file '{video_path}' not found.")
        return
    
    # Load the YOLO model
    model_path = '/workspaces/PotHole/pothole-detection-yolo11/models/best.pt'
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        return
    
    model = YOLO(model_path)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video '{video_path}'.")
        return
    
    # Get video details
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Try 'XVID' if 'mp4v' doesn't work
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    if not out.isOpened():
        print(f"Error: Could not initialize VideoWriter for '{output_path}'.")
        cap.release()
        return
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break
        
        frame_count += 1
        print(f"Processing frame {frame_count}/{total_frames}...")
        
        # Run detection
        try:
            results = model.predict(frame, imgsz=640, conf=0.5)
        except Exception as e:
            print(f"Error during detection: {e}")
            continue
        
        # Draw bounding boxes and labels
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = box.conf[0]
                label = r.names[int(box.cls[0])]
                color = (0, 255, 0)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Write the frame to the output video
        out.write(frame)

    # Release resources
    cap.release()
    out.release()
    print(f"Output saved to '{output_path}'.")

if __name__ == "__main__":
    video_path = '/workspaces/PotHole/video_2025-03-18_17-21-26.mp4'
    output_path = 'pothole-detection-yolo11/data/videos/output/output.mp4'
    detect_video(video_path, output_path)