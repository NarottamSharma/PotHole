import os
from flask import Flask, request, render_template, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename
import cv2
from ultralytics import YOLO

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './app/static/uploads'
app.config['OUTPUT_FOLDER'] = './app/static/outputs'

# Load YOLO model
model = YOLO('/workspaces/PotHole/pothole-detection-yolo11/models/best.pt')

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Home Route
@app.route('/')
def index():
    return render_template('index.html')

# Process Image
@app.route('/detect/image', methods=['POST'])
def detect_image():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # Run YOLO detection
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
    
    output_filepath = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    cv2.imwrite(output_filepath, frame)
    
    return redirect(url_for('result', filename=filename))

# Process Video
@app.route('/detect/video', methods=['POST'])
def detect_video():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    cap = cv2.VideoCapture(filepath)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    output_filepath = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    out = cv2.VideoWriter(output_filepath, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    
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
    
    return redirect(url_for('result', filename=filename))

# Display Result
@app.route('/result/<filename>')
def result(filename):
    return render_template('result.html', filename=filename)

# Serve static files
@app.route('/static/outputs/<filename>')
def output_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
