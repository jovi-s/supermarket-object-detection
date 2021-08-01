import os
import json
import shutil
import numpy as np
from tqdm import tqdm

def convert_coco_json(json_file, image_path, output_path, mode='train'):
    # Import json
    with open(json_file) as f:
        data = json.load(f)
    # Create image dict
    images = {int(x['id']): x for x in data['images']}

    # Write labels file
    for x in tqdm(data['annotations'], desc=f'Annotations {json_file}'):
        img = images[int(x['image_id'])]
        h, w, f = img['height'], img['width'], img['file_name']

        # If image not yet in output folder, copy image over
        if not os.path.isfile(output_path + 'images/' + mode + '/' + f):
            shutil.copy(image_path + f, output_path + 'images/' + mode + '/' + f)

        # The COCO box format is [top left x, top left y, width, height]
        box = np.array(x['bbox'], dtype=np.float64)
        box[:2] += box[2:] / 2  # xy top-left corner to center
        box[[0, 2]] /= w  # normalize x
        box[[1, 3]] /= h  # normalize y

        # Write to text file
        if box[2] > 0 and box[3] > 0:  # if w > 0 and h > 0
            cls = x['super_id']  # class
            line = cls, *(box)  # cls, box
            with open(output_path + 'labels/' + mode + '/' + f.split('.')[0] + '.txt', 'a') as file:
                file.write(('%g ' * len(line)).rstrip() % line + '\n')

image_path = 'D:/mvtec_d2s/images/'
coco_path = 'D:/mvtec_d2s/annotations/'
coco_filenames = ['D2S_training_super.json', 'D2S_augmented_super.json', 'D2S_validation_super.json']
yolo_path = 'D:/mvtec_yolo/'

for filename, mode in zip(coco_filenames, ['train', 'train', 'val']):
    convert_coco_json(coco_path + filename, image_path, yolo_path, mode=mode)
