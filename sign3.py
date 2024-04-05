import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Function to preprocess the dataset
def preprocess_data(train_path, test_path, img_size=(64, 64)):
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []

    label_mapping = {chr(ord('A') + i): i for i in range(26)}  # Map characters to integers

    # Process training data
    for label in os.listdir(train_path):
        label_path = os.path.join(train_path, label)
        if label in label_mapping:
            for image_file in os.listdir(label_path):
                image_path = os.path.join(label_path, image_file)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, img_size)
                image = image / 255.0  # Normalize pixel values
                train_images.append(image)
                train_labels.append(label_mapping[label])

    # Process testing data
    for label in os.listdir(test_path):
        label_path = os.path.join(test_path, label)
        if label in label_mapping:
            for image_file in os.listdir(label_path):
                image_path = os.path.join(label_path, image_file)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, img_size)
                image = image / 255.0  # Normalize pixel values
                test_images.append(image)
                test_labels.append(label_mapping[label])

    return (
        np.array(train_images),
        keras.utils.to_categorical(train_labels, num_classes=26),
        np.array(test_images),
        keras.utils.to_categorical(test_labels, num_classes=26)
    )

# Load and preprocess the dataset
train_path = "archive/asl_alphabet_train/asl_alphabet_train"
test_path = "archive/asl_alphabet_test/asl_alphabet_test"
X_train, y_train, X_test, y_test = preprocess_data(train_path, test_path)

# Model architecture
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(26, activation='softmax')  # 26 classes for A-Z
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model training
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

# User Interface Development: Assume you are using OpenCV for simplicity
cap = cv2.VideoCapture(0)  # Open default camera
while True:
    ret, frame = cap.read()
    
    # Preprocess the frame
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_resized = cv2.resize(frame_gray, (64, 64))
    frame_normalized = frame_resized / 255.0
    input_data = np.expand_dims(np.expand_dims(frame_normalized, axis=0), axis=-1)

    # Real-time prediction
    prediction = model.predict(input_data)
    predicted_label = chr(ord('A') + np.argmax(prediction))  # Convert numerical label to character
    
    # Display the result on the frame
    cv2.putText(frame, predicted_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Sign Language Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
