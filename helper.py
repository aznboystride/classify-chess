import os
import numpy as np
from keras.utils import to_categorical
import cv2, tqdm

def get_label(path):
    piece = os.path.basename(path).split('_')[0]
    d = {'0': 0, 'p': 1, 'n': 2, 'b': 3, 'r': 4, 'q': 5, 'k': 6,\
            'P': 7, 'N': 8, 'B': 9, 'R': 10, 'Q': 11, 'K': 12}
    return d[piece]

def get_img(path):
    return cv2.resize(cv2.imread(path), (224,224))

def get_data(path_dir):
    images = []
    labels = []
    for root, dirs, files in tqdm.tqdm(os.walk(path_dir)):
        for file in files:
            if 'jpg' not in file: continue
            path = os.path.join(root, file)
            image = get_img(path)
            label = get_label(path)
            images.append(image)
            labels.append(label)
    images = np.array(images)/255
    labels = to_categorical(labels, 13)
    return images, labels


