import cv2
from matplotlib.pyplot import axis
import numpy as np
from keras.models import load_model
import tensorflow as tf
from tensorflow.keras.preprocessing import image

cv2.startWindowThread()
result = {0: 'Has Mask!', 1: 'No Mask!'}
GR_dict={0:(0,255,0),1:(0,0,255)}

def image_processing(face: np.ndarray) -> tf.keras.applications.resnet50.preprocess_input:
    resize_image =  cv2.resize(face, (224,224))
    img_array = image.img_to_array(resize_image)
    img_dim_array = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.resnet50.preprocess_input(img_dim_array)

def pred_classes(preprocessed_img: np.ndarray) -> int:
    return model.predict(preprocessed_img).argmax()

model = load_model("Model/mask_detection.h5")

rect_size = 4
cap = cv2.VideoCapture("/dev/video0")


haarcascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    (rval, im) = cap.read()
    im= cv2.flip(im,1,1) 

    rerect_size = cv2.resize(im, (im.shape[1] // rect_size, im.shape[0] // rect_size))
    faces = haarcascade.detectMultiScale(rerect_size)
    for f in faces:
        (x, y, w, h) = [v * rect_size for v in f] 
        
        face_img = im[y:y+h, x:x+w]
        
        prepare_img = image_processing(face_img)
        pred = pred_classes(prepare_img)
        
        cv2.rectangle
        cv2.rectangle(im,(x,y),(x+w,y+h),GR_dict[pred],2)
        cv2.rectangle(im,(x,y-40),(x+w,y),GR_dict[pred],-1)
        cv2.putText(im, result[pred], (x+10, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

    cv2.imshow('Mask Face Detection',   im)
    # To prevent timeout
    key = cv2.waitKey(10)
    
    if key == 27: 
        break

cap.release()

cv2.destroyAllWindows()