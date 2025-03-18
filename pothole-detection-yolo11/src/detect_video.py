from ultralytics import YOLO
import cv2

def detect_video(video_path, output_path):
    model = YOLO('/workspaces/PotHole/pothole-detection-yolo11/models/best.pt')
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video '{video_path}'.")
        return
    
    # Get video details
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Define video writer
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection
        results = model.predict(frame, imgsz=640, conf=0.5)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = box.conf[0]
                label = r.names[int(box.cls[0])]
                color = (0, 255, 0)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        out.write(frame)

    cap.release()
    out.release()
    print(f"Output saved to '{output_path}'.")

if __name__ == "__main__":
    video_path = '../data/videos/test_video.mp4'
    output_path = 'pothole-detection-yolo11/data/videos/output/output.mp4'
    detect_video(video_path, output_path)
