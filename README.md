# 😷 Face Mask Detector

A real-time **Face Mask Detection System** built using **Python, OpenCV, TensorFlow/Keras, and Flask**.

The system detects faces through a webcam and classifies whether a person is wearing a mask or not. It also displays a bounding box, confidence score, generates alerts for no-mask detection, saves screenshots, and maintains detection records.

---

## 📖 Introduction

Face Mask Detector is a **Computer Vision and Deep Learning** based application that identifies whether a person is wearing a face mask.

The system uses **OpenCV** for face detection and **TensorFlow/Keras** for mask classification. A **Flask web application** provides an interface for running the detection system.

The detector classifies faces into two categories:

* 😷 **Mask**
* ❌ **No Mask**

When a face is detected, the system places a bounding box around the face and displays the predicted class along with the confidence score.

---

## 🚀 Features

* 🎥 Real-time webcam face detection
* 😷 Mask / No Mask classification
* 📦 Bounding boxes around detected faces
* 📊 Confidence score display
* 🔔 Alert when a person is detected without a mask
* 📸 Screenshot saving for violations
* 📝 Detection records and logging
* 🌐 Flask-based web interface
* 🕒 Timestamp for detection records
* 👥 Multiple face detection
* ⚡ Real-time computer vision processing

---

## 🛠️ Technologies Used

| Technology   | Purpose                              |
| ------------ | ------------------------------------ |
| Python       | Main programming language            |
| OpenCV       | Face detection and image processing  |
| TensorFlow   | Deep learning framework              |
| Keras        | Model building and prediction        |
| Flask        | Web application and backend          |
| NumPy        | Numerical and image array processing |
| Haar Cascade | Face detection                       |
| HTML/CSS     | Web interface                        |
| CSV          | Detection record storage             |

---

## 🔍 How the System Works

The system follows these steps:

1. 🎥 Webcam captures video
2. 🔍 Face detection is performed
3. ✂️ Detected face is cropped
4. 🧹 Image preprocessing is performed
5. 🧠 Face image is passed to the TensorFlow/Keras model
6. 😷 Mask / No Mask prediction is generated
7. 📊 Confidence score is calculated
8. 📦 Bounding box and label are displayed
9. 🔔 Alert is generated for No Mask detection
10. 📸 Screenshot and detection record can be saved

### Workflow

```text
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
```

---

## 📁 Project Structure

```text
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
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/face-mask-detector.git
```

### 2. Open the Project Folder

```bash
cd face-mask-detector
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

For Windows:

```bash
venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

Start the Flask application:

```bash
python app.py
```

After starting the server, open the following address in your browser:

```text
http://127.0.0.1:5000
```

Allow camera access when requested.

The application will start detecting faces from the webcam.

---

## 🧠 Model

The project uses a **TensorFlow/Keras deep learning model** to classify detected faces into two categories:

* 😷 **Mask**
* ❌ **No Mask**

The model receives a preprocessed face image as input and produces a prediction with a confidence score.

### Example Prediction

```text
Mask     : 96.52%
No Mask  : 3.48%
```

---

## 🧹 Image Preprocessing

Before sending the detected face to the model, the image goes through the following preprocessing steps:

1. Face cropping
2. Image resizing
3. Pixel normalization
4. Conversion into the required input shape
5. Model prediction

This preprocessing makes the input compatible with the trained TensorFlow/Keras model.

---

## 📊 Detection Output

The system displays a bounding box around the detected face with the predicted label and confidence score.

### Mask Detection

```text
┌───────────────────────────────┐
│                               │
│       😷 MASK                 │
│       Confidence: 96.52%      │
│                               │
└───────────────────────────────┘
```

### No Mask Detection

```text
┌───────────────────────────────┐
│                               │
│       ❌ NO MASK              │
│       Confidence: 94.21%      │
│                               │
└───────────────────────────────┘
```

An alert can also be generated when a person is detected without a mask.

---

## 📸 Screenshots

Add your actual project screenshots in the `screenshots/` folder.

### 😷 Mask Detection

```markdown
![Mask Detection](screenshots/mask-detection.png)
```

### ❌ No Mask Detection

```markdown
![No Mask Detection](screenshots/no-mask-detection.png)
```

### 🌐 Flask Web Interface

```markdown
![Flask Interface](screenshots/flask-interface.png)
```

> Replace the image names above with your actual screenshot file names.

---

## 📹 Project Demo

You can add a project demonstration video or GIF here.

```markdown
[▶️ Watch Project Demo](YOUR-DEMO-LINK)
```

Replace `YOUR-DEMO-LINK` with your actual YouTube, Google Drive, or other demo link.

---

## 🗃️ Dataset

The model uses images belonging to two classes:

```text
dataset/
├── with_mask/
└── without_mask/
```

The dataset is divided into training and validation/testing data before model training.

---

## 🔄 Project Workflow

```text
Collect Dataset
      ↓
Organize Images
      ↓
Preprocess Images
      ↓
Train Deep Learning Model
      ↓
Save Trained Model
      ↓
Start Webcam
      ↓
Detect Face
      ↓
Predict Mask / No Mask
      ↓
Display Bounding Box
      ↓
Generate Alert
      ↓
Save Screenshot / Detection Record
```

---

## 💡 Applications

This project can be used for:

* 🏢 Office entrance monitoring
* 🏭 Industrial safety monitoring
* 🏥 Healthcare environments
* 🏫 Educational institutions
* 🚉 Public transportation areas
* 🛍️ Shopping malls
* 🏬 Commercial buildings
* 🚪 Automated entry monitoring

---

## ✅ Advantages

* ⚡ Real-time detection
* 🎯 Easy to use
* 🤖 Automated monitoring
* 👁️ Reduces the need for manual monitoring
* 📊 Provides confidence scores
* 👥 Can detect multiple faces
* 📝 Maintains detection records
* 🌐 Provides a web-based interface using Flask

---

## ⚠️ Limitations

* Detection accuracy depends on camera quality.
* Poor lighting can affect face detection.
* Side-facing faces may be harder to detect.
* Very small faces may not be detected correctly.
* Partially hidden faces may reduce detection accuracy.
* Model performance depends on the quality and diversity of the training dataset.

---

## 🔮 Future Scope

The project can be improved by adding:

* 📱 Mobile application support
* ☁️ Cloud-based monitoring
* 📊 Detection analytics dashboard
* 👤 Face recognition
* 📧 Email notifications
* 📲 SMS notifications
* 🎥 CCTV camera integration
* 🗄️ Database storage
* 🤖 YOLO-based object detection
* 📈 Real-time statistics and reports

---

## 📦 Requirements

The main Python libraries required for this project include:

```text
opencv-python
tensorflow
keras
numpy
flask
```

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

## 🧪 Testing

| Test Condition           | Expected Result         |
| ------------------------ | ----------------------- |
| Person wearing mask      | 😷 Mask                 |
| Person without mask      | ❌ No Mask               |
| Multiple people          | 👥 Multiple detections  |
| Low confidence detection | 📊 Confidence displayed |
| No face present          | No detection            |

---

## 🔐 Privacy Note

This project is intended for **educational and demonstration purposes**.

If deployed in a real-world environment, appropriate privacy, consent, data-retention, and applicable legal requirements should be considered.

---

## 📄 License

This project is available for educational purposes.

You can add an **MIT License** to the repository if you want to distribute the project under the MIT License.

---

## 👨‍💻 Author

### Sangam Saini

**B.C.A – Artificial Intelligence & Machine Learning**

### Skills Demonstrated

* Python
* Computer Vision
* Machine Learning
* Deep Learning
* TensorFlow/Keras
* OpenCV
* Flask
* Git & GitHub

---

## ⭐ Support

If you find this project useful, please consider giving the repository a ⭐ on GitHub.

**Thank you for visiting this project! 😷🚀**

