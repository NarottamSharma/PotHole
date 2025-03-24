import os
import signal
import subprocess
from flask import Flask, request, render_template, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename
import cv2
from ultralytics import YOLO
from flask_cors import CORS
import logging

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Function to set request timeout
def handler(signum, frame):
    raise TimeoutError("Request took too long!")

# Set timeout (e.g., 300 seconds)
signal.signal(signal.SIGALRM, handler)

@app.before_request
def before_request():
    signal.alarm(300)  # Set a timeout of 300 seconds for each request

@app.after_request
def after_request(response):
    signal.alarm(0)  # Disable alarm after request is completed
    return response

BASE_DIR = "/workspaces/PotHole/pothole-detection-yolo11/app/static"
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, "uploads")
app.config['RAW_VIDEO_FOLDER'] = os.path.join(BASE_DIR, "raw_videos")
app.config['PROCESSED_VIDEO_FOLDER'] = os.path.join(BASE_DIR, "outputs")
app.config['CONVERTED_VIDEO_FOLDER'] = os.path.join(BASE_DIR, "converted_videos")

model = YOLO('/workspaces/PotHole/pothole-detection-yolo11/models/best.pt')

# Ensure directories exist
for folder in [app.config['UPLOAD_FOLDER'], app.config['RAW_VIDEO_FOLDER'], app.config['PROCESSED_VIDEO_FOLDER'], app.config['CONVERTED_VIDEO_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect/image', methods=['POST'])
def detect_image():
    if 'file' not in request.files:
        logging.error("No file part in the request")
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        logging.error("No selected file")
        return redirect(url_for('index'))

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Process the image using YOLO
    frame = cv2.imread(filepath)
    results = model.predict(frame, imgsz=640, conf=0.5)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0]
            label = r.names[int(box.cls[0])]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    output_filepath = os.path.join(app.config['PROCESSED_VIDEO_FOLDER'], filename)
    cv2.imwrite(output_filepath, frame)

    return redirect(url_for('result', filename=filename))

@app.route('/detect/video', methods=['POST'])
def detect_video():
    if 'file' not in request.files:
        logging.error("No file part in the request")
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        logging.error("No selected file")
        return redirect(url_for('index'))

    filename = secure_filename(file.filename)
    raw_filepath = os.path.join(app.config['RAW_VIDEO_FOLDER'], filename)
    file.save(raw_filepath)

    # Process the video with YOLO
    cap = cv2.VideoCapture(raw_filepath)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    processed_filepath = os.path.join(app.config['PROCESSED_VIDEO_FOLDER'], filename)
    out = cv2.VideoWriter(processed_filepath, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, imgsz=640, conf=0.5)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = box.conf[0]
                label = r.names[int(box.cls[0])]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()

    # Convert the processed video using FFmpeg
    converted_filename = filename.rsplit('.', 1)[0] + ".mp4"
    converted_filepath = os.path.join(app.config['CONVERTED_VIDEO_FOLDER'], converted_filename)

    ffmpeg_command = f"ffmpeg -y -i {processed_filepath} -c:v libx264 -preset slow -crf 23 -c:a aac -b:a 128k {converted_filepath}"

    try:
        subprocess.run(ffmpeg_command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg Error: {e}")
        return f"FFmpeg Error: {e}", 500

    if not os.path.exists(converted_filepath):
        logging.error(f"FFmpeg conversion failed. Video file not found: {converted_filepath}")
        return "FFmpeg conversion failed. Video file not found.", 500

    return redirect(url_for('result', filename=converted_filename))

@app.route('/result/<filename>')
def result(filename):
    return render_template('result.html', filename=filename)

@app.route('/converted_videos/<filename>')
def converted_video(filename):
    converted_filepath = os.path.join(app.config['CONVERTED_VIDEO_FOLDER'], filename)
    logging.info(f"Attempting to access video at: {converted_filepath}")
    if not os.path.exists(converted_filepath):
        logging.error(f"Video not found at path: {converted_filepath}")
        return "Converted video not found", 404
    return send_from_directory(app.config['CONVERTED_VIDEO_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
