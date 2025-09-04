import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# Load Mediapipe Hand Model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Load Trained Sign Language Model (If available)
model = tf.keras.models.load_model("asl_model.h5")  # Make sure to train a model or use a pre-trained one

# Define ASL classes (A-Z)
classes = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

# Open Webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip and Convert Frame to RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect Hand Landmarks
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract Hand Landmark Coordinates
            landmark_list = []
            for landmark in hand_landmarks.landmark:
                landmark_list.append(landmark.x)
                landmark_list.append(landmark.y)
            
            # Convert to NumPy Array
            landmark_array = np.array(landmark_list).reshape(1, -1)

            # Predict Sign Language Gesture
            prediction = model.predict(landmark_array)
            predicted_class = np.argmax(prediction)
            asl_letter = classes[predicted_class]

            # Display Detected ASL Letter
            cv2.putText(frame, f"Detected: {asl_letter}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show Webcam Feed
    cv2.imshow("Sign Language Translator", frame)

    # Exit on 'q' Key Press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release Resources
cap.release()
cv2.destroyAllWindows()
