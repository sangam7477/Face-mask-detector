📖 Introduction

Face Mask Detector is a Computer Vision and Deep Learning based application that identifies whether a person is wearing a face mask.

The system uses OpenCV for face detection and TensorFlow/Keras for mask classification. A Flask web application provides an interface for running the detection system.

The detector classifies faces into two categories:

😷 Mask
❌ No Mask

When a face is detected, the system places a bounding box around the face and displays the predicted class along with the confidence score.

🚀 Features
🎥 Real-time webcam face detection
😷 Mask / No Mask classification
📦 Bounding boxes around detected faces
📊 Confidence score display
🔔 Alert when a person is detected without a mask
📸 Screenshot saving for violations
📝 Detection records/logging
🌐 Flask-based web interface
🕒 Timestamp for detection records
👥 Supports detection of multiple faces
⚡ Real-time computer vision processing
🛠️ Technologies Used
Technology	Purpose
Python	Main programming language
OpenCV	Face detection and image processing
TensorFlow	Deep learning framework
Keras	Model building and prediction
Flask	Web application/backend
NumPy	Numerical and image array processing
Haar Cascade	Face detection
HTML/CSS	Web interface
CSV	Detection record storage
🔍 How the System Works

The system follows these steps:

Webcam
   ↓
Capture Video Frame
   ↓
Face Detection
   ↓
Crop Detected Face
   ↓
Image Preprocessing
   ↓
TensorFlow/Keras Model
   ↓
Mask / No Mask Prediction
   ↓
Confidence Score
   ↓
Bounding Box + Label
   ↓
Alert / Screenshot / Log
📁 Project Structure
face-mask-detector/
│
├── dataset/
│   ├── with_mask/
│   └── without_mask/
│
├── models/
│   ├── mask_detector_model.h5
│   ├── labels.json
│   └── model_config.json
│
├── utils/
│   └── mask_detector.py
│
├── logs/
│   └── attendance.csv
│
├── screenshots/
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
├── train_model.py
├── app.py
├── detect_mask.py
├── requirements.txt
└── README.md
⚙️ Installation
1. Clone the Repository
git clone https://github.com/YOUR-USERNAME/face-mask-detector.git
2. Open the Project Folder
cd face-mask-detector
3. Create a Virtual Environment
python -m venv venv
4. Activate Virtual Environment

For Windows:

venv\Scripts\activate
5. Install Dependencies
pip install -r requirements.txt
▶️ How to Run

Start the Flask application:

python app.py

After starting the server, open:

http://127.0.0.1:5000

Allow camera access when requested.

The application will start detecting faces from the webcam.

🧠 Model

The project uses a TensorFlow/Keras deep learning model to classify detected faces into:

Mask
No Mask

The model receives a preprocessed face image as input and produces a prediction with a confidence score.

Example:

Mask     : 96.52%
No Mask  : 3.48%
🧹 Image Preprocessing

Before sending the detected face to the model, the image goes through preprocessing steps such as:

Face cropping
Image resizing
Pixel normalization
Conversion into the required input shape
Model prediction

This makes the input compatible with the trained TensorFlow/Keras model.

📊 Detection Output

The system displays:

┌───────────────────────────────┐
│                               │
│       Face Detected           │
│       😷 MASK                 │
│       Confidence: 96.52%      │
│                               │
└───────────────────────────────┘

For a person without a mask:

NO MASK
Confidence: 94.21%

An alert can also be generated for a No Mask detection.

📸 Screenshots

Add your actual project screenshots here:

![Face Mask Detection](screenshots/mask-detection.png)
![No Mask Detection](screenshots/no-mask-detection.png)

You can also add a screenshot of your Flask web interface.

📹 Project Demo

Add your project demonstration video or GIF here:

[▶️ Watch Project Demo](YOUR-DEMO-LINK)
🗃️ Dataset

The model is trained using images belonging to two classes:

with_mask/
without_mask/

The dataset is divided into training and validation/testing data before model training.

🔄 Project Workflow
1. Collect Dataset
       ↓
2. Organize Images
       ↓
3. Preprocess Images
       ↓
4. Train Deep Learning Model
       ↓
5. Save Trained Model
       ↓
6. Start Webcam
       ↓
7. Detect Face
       ↓
8. Predict Mask / No Mask
       ↓
9. Display Bounding Box
       ↓
10. Generate Alert & Save Record
💡 Applications

This project can be used for:

🏢 Office entrance monitoring
🏭 Industrial safety monitoring
🏥 Healthcare environments
🏫 Educational institutions
🚉 Public transportation areas
🛍️ Shopping malls
🏬 Commercial buildings
🚪 Automated entry monitoring
✅ Advantages
Real-time detection
Easy to use
Automated monitoring
Reduces manual monitoring
Provides confidence scores
Can detect multiple faces
Can maintain detection records
Web-based interface using Flask
⚠️ Limitations
Detection accuracy depends on the quality of the camera.
Poor lighting can affect face detection.
Side-facing faces may be harder to detect.
Very small or partially hidden faces may not be detected correctly.
Model performance depends on the training dataset.
🔮 Future Scope

The project can be improved by adding:

📱 Mobile application support
☁️ Cloud-based monitoring
📊 Detection analytics dashboard
👤 Face recognition
📧 Email notifications
📲 SMS notifications
🎥 CCTV camera integration
🗄️ Database storage
🤖 YOLO-based object detection
📈 Real-time statistics and reports
📦 Requirements

Example requirements.txt:

opencv-python
tensorflow
keras
numpy
flask

Install all dependencies using:

pip install -r requirements.txt
🧪 Testing

The system can be tested under different conditions:

Test Condition	Expected Result
Person wearing mask	Mask
Person without mask	No Mask
Multiple people	Multiple detections
Low confidence detection	Confidence displayed
No face present	No detection
🔐 Privacy Note

This project is intended for educational and demonstration purposes. If deployed in a real environment, appropriate privacy, consent, data-retention, and applicable legal requirements should be considered.

📄 License

This project is available for educational purposes.

You can add an MIT License to the repository if you want to distribute the project under that license.
