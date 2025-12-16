import cv2
import numpy as np
import time
import csv
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

print("SCRIPT STARTED")

# Load trained emotion model
model = load_model("final_emotion_model.h5")
classes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Load Haar cascade face detector
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Webcam not accessible")
    exit()

emotion_log = []
start_time = time.time()

print("Webcam opened. Press ESC to exit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame capture failed")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))
            face = face.astype("float") / 255.0
            face = img_to_array(face)
            face = np.expand_dims(face, axis=0)

            # Predict emotion
            preds = model.predict(face, verbose=0)[0]
            emotion_index = np.argmax(preds)
            label = classes[emotion_index]
            confidence = int(preds[emotion_index] * 100)  # integer %

            current_time = round(time.time() - start_time, 2)
            emotion_log.append([current_time, label, confidence])

            # Draw face box and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            text = f"{label} ({confidence}%)"
            cv2.putText(frame, text, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Emotion Detection (ESC to Exit)", frame)

        if cv2.waitKey(10) & 0xFF == 27:
            print("ESC pressed. Exiting loop.")
            break

finally:
    print("FINALLY BLOCK EXECUTING")

    cap.release()
    cv2.destroyAllWindows()

    # Overwrite same CSV each run
    file_path = os.path.join(os.getcwd(), "emotion_log.csv")

    print("Saving CSV to:", file_path)
    print("Total emotion records:", len(emotion_log))

    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Time (seconds)", "Emotion", "Confidence (%)"])
        writer.writerows(emotion_log)

    print("CSV WRITE COMPLETE")
    print("PROGRAM EXITED SUCCESSFULLY")
