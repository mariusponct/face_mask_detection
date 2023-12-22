import cv2
import numpy as np
import os 
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models


def preprocess_dataset(root_folder):
    data = []
    labels = []

    for label in os.listdir(root_folder):
        path = os.path.join(root_folder, label)
        
        # Skip files that are not directories/ for mac
        if not os.path.isdir(path):
            continue

        for img_file in os.listdir(path):
            img_path = os.path.join(path, img_file)
            
            # Skip files that are not images 
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            
            img = cv2.imread(img_path)
            img = cv2.resize(img, (64, 64))  # Adjust the size as needed
            img = img / 255.0  # Normalize pixel values to [0, 1]

            data.append(img)
            labels.append(label)

    return np.array(data), np.array(labels)


# Set up  dataset
root_folder = "/Users/mariusmuresan/Desktop/face_mask_d_v3/dataset_v3/dataset"
data, labels = preprocess_dataset(root_folder)

# Build a Convolutional Neural Network (CNN) using TensorFlow
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))  # Binary classification
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Convert string labels to numerical labels
label_mapping = {"without_mask": 1, "with_mask": 0}
y_train = np.array([label_mapping[label] for label in y_train])
y_test = np.array([label_mapping[label] for label in y_test])

# Train the model
model.fit(X_train, y_train, epochs=12, validation_data=(X_test, y_test))

# Implement the face mask detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = img[y:y + h, x:x + w]
        face = cv2.resize(face, (64, 64))
        face = face / 255.0
        face = np.reshape(face, (1, 64, 64, 3))
        prediction = model.predict(face)

        if prediction < 0.7:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, 'Mask', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(img, 'No Mask', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    return img

# Capture video frames and apply face mask detection

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = detect_face(frame)

    cv2.imshow('Face Mask Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
