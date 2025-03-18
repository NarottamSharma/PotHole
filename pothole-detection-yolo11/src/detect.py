from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

def detect(image_path):
    model = YOLO('/workspaces/PotHole/pothole-detection-yolo11/models/best.pt')

    # Run inference
    results = model.predict(image_path, save=True, imgsz=640)

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0]
            label = r.names[int(box.cls[0])]
            color = (0, 255, 0)
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(img, f"{label} {conf:.2f}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.axis('off')
    plt.title('Pothole Detection Result')
    plt.show()

if __name__ == "__main__":
    img_path = '/workspaces/PotHole/pothole-detection-yolo11/data/test/images'
    detect(img_path)
