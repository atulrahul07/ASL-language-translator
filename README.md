ASL Language Translator 🤟
Welcome to the ASL Language Translator! This project leverages the power of computer vision and deep learning to provide real-time American Sign Language (ASL) alphabet recognition. Our goal is to create a tool that can help bridge communication gaps and make the world more accessible.

✨ Features
Real-time Recognition: Translate ASL gestures into text instantly using your webcam.

Hand Landmark Extraction: Utilizes Mediapipe to accurately detect and extract key points from the user's hand.

Deep Learning Model: A TensorFlow Multi-Layer Perceptron (MLP) model is trained to classify 24 different ASL alphabet gestures.

Automated Data Collection: A built-in script allows for easy and efficient collection of training data.

Model Training: A dedicated script to train the classification model with your collected data.

Live Prediction: A user-friendly interface to run the trained model and see live predictions.

Simple Setup: Get up and running quickly with clear instructions and minimal dependencies.

🚀 How It Works
The system follows a simple yet effective pipeline:

Capture: The webcam captures video frames.

Landmark Extraction: Mediapipe processes each frame to identify and track 21 key hand landmarks. These landmarks are represented as (x, y, z) coordinates.

Data Preprocessing: The landmark coordinates are normalized and flattened into a single feature vector.

Prediction: The feature vector is fed into our trained TensorFlow MLP model.

Classification: The model outputs a prediction for the corresponding ASL alphabet letter.

🛠️ Installation & Setup
Clone the repository:

git clone [https://github.com/atulrahul07/ASL-language-translator.git](https://github.com/atulrahul07/ASL-language-translator.git)
cd ASL-language-translator

Install dependencies:

pip install -r requirements.txt

Run the live prediction script:

python live_prediction.py

Note: The requirements.txt file contains all the necessary libraries, including tensorflow, mediapipe, and opencv-python.

📊 Automated Data Collection
If you want to train your own model, you can use the data collection script:

Create a folder named data in the project directory.

Run the data_collection.py script:

python data_collection.py

Follow the on-screen instructions to collect data for each letter.

🧠 Model Training
After collecting your data, you can train the model with the following command:

python training.py

This script will train the MLP model and save it as a file named model.h5.

🙏 Credits
This project was inspired by the work of other creators in the field of computer vision and deep learning. Special thanks to Google's Mediapipe for providing the robust hand tracking solution.

Made with ❤️ by atulrahul07
