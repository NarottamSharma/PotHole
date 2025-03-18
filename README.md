# 🚀 Pothole Detection with YOLOv11 
### 🏆 Real-time Pothole Detection Using YOLOv11 and OpenCV  
![GitHub stars](https://img.shields.io/github/stars/your-repo) ![GitHub forks](https://img.shields.io/github/forks/NarottamSharma/PotHole) ![GitHub last commit](https://img.shields.io/github/last-commit/NarottamSharma/PotHole)  

---

<div align="center">
  <img src="https://media.giphy.com/media/3o7abldj0b3rxrZUxW/giphy.gif" width="600" alt="Pothole Detection Demo">
</div>

---

## 🌟 About the Project  
This project implements a **YOLOv8-based pothole detection** system capable of processing both images and videos, along with real-time detection using OpenCV.  

✅ Real-time detection using YOLOv11 
✅ High accuracy with efficient inference  
✅ Training from scratch or using pre-trained weights  

---

## 🎯 Features  
✔️ State-of-the-art YOLOv11 model  
✔️ Fast processing with OpenCV  
✔️ Supports both image and video input  
✔️ Clean and modular code structure  
✔️ Jupyter notebook for experimentation  

---

## 🏗️ Project Structure  
```plaintext
pothole-detection-yolo11/
├── data/
│   ├── images/              # Raw images
│   ├── labels/              # YOLO-format labels
│   ├── data.yaml            # Dataset config
├── models/
│   ├── yolo11n.pt           # Pre-trained model
│   └── best.pt              # Trained model (after training)
├── src/
│   ├── train.py             # Training script
│   ├── detect.py            # Inference on images
│   ├── real_time.py         # Real-time detection script
├── notebooks/
│   └── experiment.ipynb     # Jupyter notebook for experimentation
├── requirements.txt         # Package dependencies
├── README.md                # Project documentation
