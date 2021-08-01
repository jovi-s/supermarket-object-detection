import os

import cv2
import pandas
import numpy as np
import logging
import torch
from flask import Flask, request, Response, render_template

from .camera import VideoCamera
# Initialize webcam class
video_stream = VideoCamera()

app = Flask(__name__)


# Model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model = video_stream.load_model()


@app.route('/')
def index():
    # Health check endpoint
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        logging.info(f"Uploaded image: {file.filename}")

        # save the file to /uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'static', 'uploads', file.filename)
        logging.info(f"Image uploaded to: {file_path}")
        file.save(file_path)

        # Inference
        logging.info('Running prediction on uploaded image...')
        results = model(file_path)

        save_path = os.path.join(basepath, 'static', 'uploads_detections')
        logging.info(f"Image with objects detected saved at: {save_path}")
        results.save(save_path)
        
        # Count classes
        df = results.pandas().xyxy[0]
        count = df.groupby('name')['name'].agg(['count'])
        count = count.sort_values(by=['count'], ascending=False)

        logging.info(f"Objects detected: {count.to_json()}")

        return render_template('predict.html', 
                                pred_image=file.filename,
                                results=count.to_json())

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(video_stream),
                mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    # app.run(debug=True, port=8000)
    app.run(host="127.0.0.1", debug=True, port=5000)