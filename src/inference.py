import argparse
import torch

OBJECTS = []

# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='small_100_16_640.pt')

def detect(img_path):
    results = model(img_path) # 
    # detect_classes_pandas = results.pandas().xyxy[0]
    # detect_classes = results.xyxy[0].detach().numpy()
    # detect_classes = results.pandas().xyxy[0].to_json(orient="records")
    # for result in results.xyxyn:
    #     for pred in result:
    #         return print(pred)
    # return print(detect_classes)
    return results.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', default="grocery-store.jpg")

    args = parser.parse_args()
    detect(args.img_path)