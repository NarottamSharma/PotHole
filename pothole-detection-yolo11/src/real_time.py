from ultralytics import YOLO
import cv2

# Load the trained model
model = YOLO('/workspaces/PotHole/pothole-detection-yolo11/models/best.pt')

# Open the webcam
cap = cv2.VideoCapture(1)  # 0 for default webcam

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image.")
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

    cv2.imshow("Pothole Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
