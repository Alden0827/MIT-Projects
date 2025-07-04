import cv2
import os
import uuid  # For generating random unique filenames

# 📁 Paths
SOURCE_DIR = './ss'
OUTPUT_DIR = './faces'
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# 📐 Fixed face size
FACE_WIDTH = 160
FACE_HEIGHT = 160

# ⚙️ Sensitivity Settings
SCALE_FACTOR = 1.1    # Lower = more sensitive
MIN_NEIGHBORS = 4     # Lower = more faces, Higher = fewer (more strict)

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load Haar Cascade Classifier
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# 🔄 Process all PNG images in SOURCE_DIR
for filename in os.listdir(SOURCE_DIR):
    if filename.lower().endswith('.png'):
        image_path = os.path.join(SOURCE_DIR, filename)
        print(f"🔍 Processing: {filename}")

        # Load and convert to grayscale
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=SCALE_FACTOR,
            minNeighbors=MIN_NEIGHBORS
        )

        # ✂️ Crop and save faces
        for (x, y, w, h) in faces:
            face = image[y:y+h, x:x+w]
            resized_face = cv2.resize(face, (FACE_WIDTH, FACE_HEIGHT))

            # Generate random filename
            random_filename = f"{uuid.uuid4().hex}.png"
            face_path = os.path.join(OUTPUT_DIR, random_filename)

            # Save face
            cv2.imwrite(face_path, resized_face)
            print(f"✅ Saved: {face_path}")

print("🎉 All faces extracted and saved with random filenames.")
