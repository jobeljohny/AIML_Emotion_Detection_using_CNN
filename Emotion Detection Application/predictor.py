from keras.models import model_from_json
from utils import img_to_array
import cv2
import numpy as np
model = model_from_json(open("models/emotion.json", "r").read())
model.load_weights('models/emotion.h5')
face_haar_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
emotion_detection = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')


def predict(frame):
    gray_image= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_haar_cascade.detectMultiScale(gray_image)
    try:
        for (x,y, w, h) in faces:
            #cv2.rectangle(frame, pt1 = (x,y),pt2 = (x+w, y+h), color = (255,0,0),thickness =  2)
            roi_gray = gray_image[y-5:y+h+5,x-5:x+w+5]
            roi_gray=cv2.resize(roi_gray,(48,48))
            image_pixels = img_to_array(roi_gray)
            image_pixels = np.expand_dims(image_pixels, axis = 0)
            image_pixels /= 255
            predictions = model.predict(image_pixels)
            max_index = np.argmax(predictions[0])
            
            emotion_prediction = emotion_detection[max_index]
            return emotion_prediction

    except :
            return 'no face detected'