import numpy as np
import cv2
import torch
import logging


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.classes = ['brown glass', 'coca cola', 'canned soft drink',
            'water', 'zucchinis', 'textured packaging', 'tea',
            'packaged cereal bars', 'cereals', 'bananas', 'cardboard tray',
            'plastic tray', 'ink cartridge', 'apple spritzer', 'nets',
            'cucumbers', 'pasta', 'apples', 'board eraser', 'pears',
            'avocados', 'foil', 'salad', 'carrots', 'blue glass', 'kiwis',
            'orange citrus']
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        self.font = cv2.FONT_HERSHEY_PLAIN
        self.counter = dict()

    def load_model(self):
        # self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', 
        #   pretrained=True)
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', 
            path='full_50_2_640.pt')
        return self.model

    def __del__(self):
        self.video.release()

    def get_frame(self):
        # Read next stream frame
        n = 0
        while self.video.isOpened():
            n += 1
            self.video.grab()
            # read every 4th frame
            if n == 4:
                # Capture frame-by-frame
                ret, frame = self.video.read()
                height, width, channels = frame.shape

                #Run detection
                results = self.model(frame) # detections

                for result in results.xyxyn:
                    for pred in result:
                        detect = {
                            "class": int(pred[5]),
                            "class_name": self.model.model.names[int(pred[5])],
                            "normalized_box": pred[:4].tolist(),
                            "confidence": float(pred[4]),
                        }
                        if detect["confidence"] > 0.5:
                            xmin = int(detect["normalized_box"][0] * width) 
                            ymin = int(detect["normalized_box"][1] * height)
                            xmax = int(detect["normalized_box"][2] * width)
                            ymax = int(detect["normalized_box"][3] * height)

                            # draw the prediction on the frame
                            cv2.rectangle(frame, (xmin,ymin), (xmax, ymax), 
                                (255,255,255), 2)
                            cv2.putText(frame, detect["class_name"], 
                                (xmin, ymin - 5), self.font, 1, (255,255,255), 1)

                            # increment and display counter
                            if detect["class_name"] not in self.counter:
                                self.counter[detect["class_name"]] = 1
                            else:
                                self.counter[detect["class_name"]] += 1

                    # display counts           
                    for i, key in enumerate(sorted(self.counter.keys())):
                        cv2.putText(frame, f"{key}: {self.counter[key]}", 
                            (15 , 15 + i * 15), self.font, 1, (255,255,255), 1)

                ret, jpeg = cv2.imencode('.jpg', frame)
                return jpeg.tobytes()

            else:
                cv2.waitKey(1)

            # reset counter for each frame
            self.counter = dict()