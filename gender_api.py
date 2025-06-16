#!/usr/bin/env python3
from flask import Flask, request, jsonify
import base64
# 3rd party dependencies
import cv2 as cv
import numpy as np

# project dependencies
from deepface import DeepFace
from deepface.models.demography import Age, Emotion, Gender, Race
from deepface.commons.logger import Logger
import os
logger = Logger()
import time

# import mediapipe as mp

app = Flask(__name__)

detectors = ["yolov8"]
detector = detectors[0]

os.environ['DEEPFACE_HOME']= os.getcwd()

DeepFace.analyze(
    "dataset/0_img.jpg", actions=("gender",), detector_backend=detector, enforce_detection=False
)

# Initialize MediaPipe Face Detection
# mp_face_detection = mp.solutions.face_detection
# mp_drawing = mp.solutions.drawing_utils

print("Warm up")



def detect_and_crop_face(img):
    """
    Detect face in the image using MediaPipe and crop it
    Returns: cropped face image or None if no face detected
    Expands bounding box by a factor of 1.5 while keeping within image bounds
    """

    
    h, w = img.shape[:2]
    print(f"Original image size: width:{w}, height:{h}")
    
    # Convert BGR to RGB for MediaPipe
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        # Process the image and find faces
        results = face_detection.process(img_rgb)
        
        # If no face detected
        if not results.detections:
            print("No face detected.")
            return None, img
        
        # Get the first face detected (assuming one person in the image)
        detection = results.detections[0]
        
        # Get bounding box coordinates
        bbox = detection.location_data.relative_bounding_box
        
        # Convert relative coordinates to absolute
        xmin = int(bbox.xmin * w)
        ymin = int(bbox.ymin * h)
        width = int(bbox.width * w)
        height = int(bbox.height * h)
        
        # Calculate center of the face
        center_x = xmin + width // 2
        center_y = ymin + height // 2
        
        # Expand the bounding box by a factor of 1.5
        new_width = int(width * 2.1)
        new_height = int(height * 2.1)
        
        # Calculate new coordinates based on center
        x1 = max(0, center_x - new_width // 2)
        y1 = max(0, center_y - new_height // 2)
        x2 = min(w, center_x + new_width // 2)
        y2 = min(h, center_y + new_height // 2)
        
        # Crop the face
        face_crop = img[y1:y2, x1:x2]
        
        # Debug: Draw bounding box on original image
        debug_img = img.copy()
        cv.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        return face_crop, debug_img
    
def handle_image_gender(image_data):
    try:
        print("Processing gender detection image")
        try:
            results = DeepFace.analyze(
                image_data, actions=("gender",), detector_backend=detector
            )
            result = results[0]["dominant_gender"]
            gender = 'Female' if result == 'Woman' else 'Male'
            return 0, gender
        except:
            result = None
            return 0, result
    except Exception as e:
        print(f"Gender handle error: {e}")
        return -1, "handle failed"

# ---------- 路由区域 ----------

@app.route('/image_gender', methods=['POST'])
def image_gender():
    request_time = time.time() 
    data = request.json
    image_base64 = data.get("image")
    try:
        img = base64.b64decode(image_base64, validate=True)  # `validate=True` 进行严格检查
    except base64.binascii.Error as e:
        return f"Base64 decoding error: {e}", 400
    print(f"Decoded image length: {len(img)} bytes")
    image_data = np.frombuffer(img, np.uint8) # 将字节数据转换为 numpy 数组
    #image_data = np.fromstring(img, np.uint8)
    if image_data.size == 0:
        return "Error: Empty image data after decoding", 400
    image_data = cv.imdecode(image_data, cv.IMREAD_COLOR)# <class 'numpy.ndarray'>
    if image_data is None:
        return "Error: cv.imdecode failed, possibly due to incorrect image format", 400
    
    img = cv.resize(image_data, (400, 400))
    # img = img.transpose(2, 0, 1)
    # assert img.shape == (3, image_h, image_w)
    # assert np.max(img) <= 255
    # input = torch.FloatTensor(img / 255.).unsqueeze(0).to(device)

    code, result = handle_image_gender(img)
    response_time = time.time()
    process_time = response_time-request_time
    return jsonify({
        "code": code,
        "result": result,
        "process_time": process_time
    })



if __name__ == '__main__':
    # 启动 Flask 服务器，运行在10001端口，并允许外部访问
    app.run(debug=True, host='0.0.0.0', port=10004)
