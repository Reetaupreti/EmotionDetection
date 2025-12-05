import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque

# Load emotion model
model = load_model("emotion_detector.h5")
emotion_classes = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

# Load DNN face detector
face_net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel"
)

# Initialize webcam
cam = cv2.VideoCapture(0)

# For smoothing predictions
smoothed_predictions = {}  # key: face id (if multiple faces)
SMOOTHING_BUFFER = 5       # number of frames to average

while True:
    ret, frame = cam.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    faces_current_frame = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            faces_current_frame.append((x1, y1, x2, y2))

            # Extract face and preprocess
            face = frame[y1:y2, x1:x2]
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (48, 48))
            gray = gray.reshape(1,48,48,1)/255.0

            prediction = model.predict(gray)[0]  # get array of probabilities

            # Smooth predictions using deque
            key = (x1, y1, x2, y2)  # using position as face id
            if key not in smoothed_predictions:
                smoothed_predictions[key] = deque(maxlen=SMOOTHING_BUFFER)
            smoothed_predictions[key].append(prediction)
            avg_pred = np.mean(smoothed_predictions[key], axis=0)

            emotion = emotion_classes[np.argmax(avg_pred)]
            confidence_score = avg_pred[np.argmax(avg_pred)]

            # Draw rectangle and label
            label = f"{emotion} ({confidence_score*100:.1f}%)"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    # Remove old faces from smoothing dict
    smoothed_predictions = {k: v for k, v in smoothed_predictions.items() if k in faces_current_frame}

    cv2.imshow("Smooth Emotion Detector", frame)
    if cv2.waitKey(1) == 27:  # ESC to exit
        break

cam.release()
cv2.destroyAllWindows()
