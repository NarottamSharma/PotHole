import os
from pathlib import Path

project_name = "pothole-detection-yolo11"

# Define the project structure
list_of_files = [
    f"{project_name}/data/images/.gitkeep",      # Keep empty directories in git
    f"{project_name}/data/labels/.gitkeep",
    f"{project_name}/data/data.yaml",            
    f"{project_name}/models/yolo11n.pt",         # Placeholder for pre-trained model
    f"{project_name}/models/best.pt",            # Placeholder for trained model
    f"{project_name}/src/train.py",              
    f"{project_name}/src/detect.py",
    f"{project_name}/src/detect_video.py",             
    f"{project_name}/src/real_time.py",          # Real-time detection script
    f"{project_name}/notebooks/experiment.ipynb", # Jupyter notebook for experimentation
    f"{project_name}/requirements.txt",          
    f"{project_name}/README.md"
]

# Create directories and files
for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    
    if filedir:
        os.makedirs(filedir, exist_ok=True)
        
    # Create file if it does not exist or is empty
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            # Add minimal content for Jupyter notebook
            if filepath.suffix == ".ipynb":
                notebook_content = {
                    "cells": [],
                    "metadata": {},
                    "nbformat": 4,
                    "nbformat_minor": 2
                }
                import json
                f.write(json.dumps(notebook_content, indent=4))
            pass
    else:
        print(f"File already exists: {filepath}")

print("✅ Project structure created!")
