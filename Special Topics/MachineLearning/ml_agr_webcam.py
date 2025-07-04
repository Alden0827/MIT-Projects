# pip install tensorflow opencv-python matplotlib numpy
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load the trained multi-output model
print('Loading model...')
model = load_model("age_gender_race_classifier_model.keras")

# Define labels
gender_labels = ['Male', 'Female']
race_labels = ['White', 'Black', 'Asian', 'Indian', 'Other']

# Load OpenCV's pre-trained face detector
print('Loading face detector model...')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
print(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


# Start webcam
print('Opening Camera!')
cap = cv2.VideoCapture(0)
print("[INFO] Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Extract face
        face_img = frame[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (64, 64))
        face_img = face_img.astype("float32") / 255.0
        face_img = np.expand_dims(face_img, axis=0)

        # Predict age, gender, and race
        age_pred, gender_pred, race_pred = model.predict(face_img, verbose=0)

        # Get age and map to age range
        age = int(age_pred[0][0])
        if age <= 4:
            age_range = "Infant/Toddler (0 to 4)"
        elif age <= 9:
            age_range = "Child (5 to 9)"
        elif age <= 12:
            age_range = "Pre-Teen (10 to 12)"
        elif age <= 17:
            age_range = "Teen (13 to 17)"
        elif age <= 24:
            age_range = "Young Adult (18 to 24)"
        elif age <= 35:
            age_range = "Adult I (25 to 35)"
        elif age <= 45:
            age_range = "Adult II (36 to 45)"
        elif age <= 59:
            age_range = "Middle-aged Adult (46 to 59)"
        elif age <= 69:
            age_range = "Senior I (60 to 69)"
        elif age <= 79:
            age_range = "Senior II (70 to 79)"
        else:
            age_range = "Elderly (80 and up)"



        # gender = gender_labels[np.argmax(gender_pred[0])]
        # gender_conf = np.max(gender_pred[0]) * 100
        # Adjusted threshold-based gender prediction
        gender_score = gender_pred[0][1]  # Confidence score for 'Female'

        if gender_score > 0.45:
            gender = "Female"
            gender_conf = gender_score * 100
        else:
            gender = "Male"
            gender_conf = (1 - gender_score) * 100

        
        race = race_labels[np.argmax(race_pred[0])]
        race_conf = np.max(race_pred[0]) * 100

        # Set base position for text
        x_pos = x
        y_pos = y + h + 25
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 1

        # Draw each label with different color
        cv2.putText(frame, f"{gender} ({gender_conf:.1f}%)", (x_pos, y_pos), font, scale, (255, 0, 0), thickness)     # Blue
        # cv2.putText(frame, f"Age: {age}", (x_pos, y_pos + 20), font, scale, (0, 255, 0), thickness)                   # Green
        cv2.putText(frame, f"Age: {age_range}", (x_pos, y_pos + 20), font, scale, (0, 255, 0), thickness)  # Green
        cv2.putText(frame, f"Skin Color: {race} ({race_conf:.1f}%)", (x_pos, y_pos + 40), font, scale, (0, 0, 255), thickness) # Red

    # Display webcam frame
    cv2.imshow("Age, Gender, Race Classifier", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
