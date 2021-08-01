import os
import shutil
import random

def subsample(main_path, sub_path, min_size=10, max_size=20, num_classes=27, seed=42):
    random.seed(seed)
    train_image_names = os.listdir(main_path + 'images/train/')
    val_image_names = os.listdir(main_path + 'images/val/')
    for mode, filenames in zip(['train', 'val'], [train_image_names, val_image_names]):
        counts = [0] * num_classes
        selected = set()
        copied = set()
        while min(counts) < min_size:
            # Select a random image file
            filename = random.choice(filenames)
            while filename in selected:
                filename = random.choice(filenames)
                selected.add(filename)
            # Access the correspinding label file
            label_filename = 'labels/' + mode + '/' + filename.split('.')[0] + '.txt'
            super_ids = []
            with open(main_path + label_filename, 'r') as f:
                # Count the object instances in the file
                for line in f:
                    super_ids.append(int(line.split()[0])) 
            # If mode is 'train' and any category already exceeded the max_size, skip the image
            exceeded = False
            if mode == 'train':
                for idx in super_ids:
                    if counts[idx] >= max_size:
                        exceeded = True
                        break
            if not exceeded:
                copied.add(filename)
                for idx in super_ids:
                    counts[idx] += 1
                # Copy selected image and label file to sub_path folder
                shutil.copy(main_path + 'images/' + mode + '/' + filename,  # source
                            sub_path + 'images/' + mode + '/' + filename)   # destination
                shutil.copy(main_path + label_filename,  # source
                            sub_path + label_filename)   # destination
        print(len(copied), counts)

main_path = 'D:/mvtec_yolo/'
sub_path = 'D:/mvtec_yolo_small/'

subsample(main_path, sub_path, min_size=15, max_size=60)
