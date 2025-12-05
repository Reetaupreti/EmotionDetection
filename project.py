import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model & face detector
model = load_model("emotion_detector.h5")
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

emotion_classes = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48,48))
        face = face.reshape(1,48,48,1) / 255.0

        prediction = model.predict(face)
        emotion = emotion_classes[np.argmax(prediction)]

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame, emotion,(x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),2)

    cv2.imshow("Emotion Detector", frame)
    if cv2.waitKey(1)==27:   # Press ESC to exit
        break

cam.release()
cv2.destroyAllWindows()
