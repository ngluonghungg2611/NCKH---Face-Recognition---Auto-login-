import numpy as np
import json
import cv2
import torch
from PIL import Image



def _crop_face_224(img, img_path, img_name):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    img = cv2.imread(img_path + img_name)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(img_gray, scaleFactor = 1.5, minNeighbors = 5)
    (x, y, w, h) = face[0,:]
    img_face = img[y: y + h, x: x + w]
    img_face = cv2.resize(img_face, (224,224))
    cv2.imwrite(img_path + img_name, img_face)
    
class NumpyEncoder(json.JSONEncoder):
    '''
    Encoding numpy into json
    '''
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.int32):
            return int(obj)
        if isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, np.float64):
            return float(obj)
        return json.JSONEncoder.default(self, obj)