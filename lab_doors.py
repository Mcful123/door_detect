# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 15:02:03 2021

@author: chomi
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import cv2 as cv

np.set_printoptions(suppress=True)
xrd_model = tensorflow.keras.models.load_model('xrd.h5', compile=False)
furn_model = tensorflow.keras.models.load_model('furnace.h5', compile=False)

def pred(im, mod):
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    size = (224, 224)
    image = im
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    prediction = mod.predict(data)
    if(prediction[0][0] > 0.6):
        return 'OPEN'
    elif(prediction[0][1] > 0.6):
        return 'CLOSED'
    else:
        return 'NA'

def convert(cv_im):
    cv_im = cv.resize(cv_im, (224,224), interpolation = cv.INTER_AREA)
    cv_im2 = cv.cvtColor(cv_im, cv.COLOR_BGR2RGB)
    pil_im  = Image.fromarray(cv_im2)
    return pil_im

def detect(frame):
    xrd_door = frame[190:420, 480:638]
    furn_door = frame[79:238, 144:224]
    
    xrd_door_PIL = convert(xrd_door)
    furn_door_PIL = convert(furn_door)
    
    xrd_status = pred(xrd_door_PIL, xrd_model)
    furn_status = pred(furn_door_PIL, furn_model)
    
    statuses = {'XRD':xrd_status, 'Furnace': furn_status}
    return statuses
    
if(__name__ == "__main__"):
    vid = cv.VideoCapture(0, cv.CAP_DSHOW) 
    while(True):
        _, frame = vid.read()
        if(frame[0][0][0] == None):
            continue
        statuses = detect(frame)
        cv.imshow('cam', frame)
        
        key = cv.waitKey(1)
        if(key == ord('q')):
            break
        elif(key == ord('s')):
            print(statuses)
            
    vid.release()
    cv.destroyAllWindows()


