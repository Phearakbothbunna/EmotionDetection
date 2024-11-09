import cv2
import numpy as np
import tensorflow as tf
from prepare_data import reverse_label_map

# Load the trained emotion recognition model
model = tf.keras.models.load_model('emotion_cnn_model.keras')

# Emotion label mapping
label_map = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'sad': 4, 'surprise': 5, 'neutral': 6}
reverse_label_map = {v: k for k, v in label_map.items()}

# Initialize the webcam (0 is the default webcam)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the region of interest
        roi_gray = gray[y:y+h, x:x+w]

        # Resize and normalize the image
        resized_face = cv2.resize(roi_gray, (48, 48))
        normalized_face = resized_face / 255.0

        # Prepare the image for the model
        input_face = np.expand_dims(normalized_face, axis=-1)  # Shape: (48, 48, 1)
        input_face = np.expand_dims(input_face, axis=0)  # Shape: (1, 48, 48, 1)

        # Make a prediction
        prediction = model.predict(input_face)
        predicted_class = np.argmax(prediction, axis=1)[0]
        predicted_emotion = reverse_label_map[predicted_class]

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Emotion Recognition', frame)

    # Break on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close windows
cap.release()
cv2.destroyAllWindows()
