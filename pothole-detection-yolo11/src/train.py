from ultralytics import YOLO
import torch
import os

# Clear CUDA cache to prevent memory issues
torch.cuda.empty_cache()

MODEL_PATH = '/workspaces/PotHole/pothole-detection-yolo11/models/best.pt'

def train():
    if os.path.exists(MODEL_PATH):
        print(f"Model already trained. Found at '{MODEL_PATH}'. Skipping training.")
        return
    
    model = YOLO('/workspaces/PotHole/pothole-detection-yolo11/models/yolo11n.pt')  # Load pre-trained model

    # Train the model
    results = model.train(
        data='../data/data.yaml',
        epochs=100,
        imgsz=640,
        batch=4,
        device=0,                # GPU (set to 'cpu' if no GPU available)
        workers=0,
        half=True,
        lr0=0.01,
        lrf=0.1,
        weight_decay=0.0005,
        patience=20,
        augment=False
    )

    # Save the trained model
    model.save(MODEL_PATH)
    print(f"Model training complete. Best model saved to '{MODEL_PATH}'.")

if __name__ == "__main__":
    train()
