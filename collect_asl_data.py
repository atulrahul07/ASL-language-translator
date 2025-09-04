import cv2
import mediapipe as mp
import pandas as pd
import os
import time

# Initialize Mediapipe Hand Tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Open the webcam
cap = cv2.VideoCapture(0)

# Label for the ASL letter you are collecting
label = input("Enter the ASL letter you are collecting data for: ").upper()

# Ensure dataset file exists
csv_filename = "asl_data.csv"
if not os.path.exists(csv_filename):
    with open(csv_filename, "w") as f:
        f.write(",".join([f"landmark_{i}" for i in range(21*2)]) + ",label\n")  # 21 landmarks * 2 (x, y)

# Data collection settings
num_samples = 200  # Set how many samples to collect
collected = 0

print(f"📸 Collecting {num_samples} samples for letter '{label}'. Press 'q' to quit.")

while collected < num_samples:
    ret, frame = cap.read()
    if not ret:
        continue

    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands
    results = hands.process(rgb_frame)

    # If hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract landmark positions
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)

            # Save data
            with open(csv_filename, "a") as f:
                f.write(",".join(map(str, landmarks)) + f",{label}\n")

            collected += 1
            print(f"✅ Collected {collected}/{num_samples} samples")

            # Draw landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Show the frame
    cv2.imshow("Data Collection", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

print("✅ Data collection complete!")
cap.release()
cv2.destroyAllWindows()
