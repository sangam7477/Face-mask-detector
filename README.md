# 😷 Face Mask Detector

A real-time **Face Mask Detection System** built using **Python, OpenCV, and TensorFlow/Keras**. The system detects faces through a webcam and classifies them as **Mask** or **No Mask**.

It also displays confidence scores, draws labeled bounding boxes, plays an alert sound for no-mask detections, saves violation screenshots, and maintains running detection totals.

## 📁 Project Structure

```text
face-mask-detector/
│
├── dataset/
│   ├── with_mask/
│   └── without_mask/
│
├── models/
│
├── utils/
│
├── app.py
├── train_model.py
├── detect_mask.py
└── requirements.txt
```

## ⚙️ Installation

First, clone or download the project and open the project folder in your terminal.

Install all required Python packages:

```bash
pip install -r requirements.txt
```

> **Note:** If you are using Apple Silicon (M1/M2/M3/M4), you may need to install `tensorflow-macos` instead of the standard TensorFlow package.

---

## 🧠 1. Train the Model

Place your training images inside the following folders:

```text
dataset/
├── with_mask/
└── without_mask/
```

The repository keeps these folders as placeholders to keep the GitHub repository lightweight. Add your own training images before training or retraining the model.

Run the training script:

```bash
python train_model.py --dataset dataset --epochs 15 --batch-size 32
```

### Training Outputs

After successful training, the following files will be generated:

```text
models/
├── mask_detector_model.h5
└── labels.json
```

---

## 🎥 2. Run Real-Time Face Mask Detection

Start the webcam-based detection system:

```bash
python detect_mask.py
```

### Available Options

Select a specific webcam:

```bash
python detect_mask.py --camera 0
```

Disable the alert sound:

```bash
python detect_mask.py --no-alert
```

Specify the screenshot directory:

```bash
python detect_mask.py --screenshot-dir static/sample_images
```

Press **Q** to stop the webcam detection.

---

## 🌐 3. Run the Flask Web Application

The project also includes an optional **Flask web application**.

Start the Flask server:

```bash
python app.py
```

Then open the following address in your browser:

```text
http://localhost:5000
```

Flask provides the web interface through which the face mask detection system can be accessed.

---

## ✨ Features

* 🎥 Real-time face detection using a webcam
* 😷 Mask vs. No Mask classification
* 📦 Bounding boxes around detected faces
* 📊 Confidence score display
* 🔴 Red bounding box for **No Mask**
* 🟢 Green bounding box for **Mask**
* 🔊 Alert sound for no-mask detection
* 📸 Automatic screenshots for violations
* 📈 Running detection totals
* ⏱️ Screenshot cooldown to avoid duplicate captures
* 🌐 Optional Flask web interface

## 📝 Notes

* **Green bounding box:** Mask detected
* **Red bounding box:** No Mask detected
* Each detected face displays its classification and confidence score.
* A screenshot is automatically saved when a **No Mask** violation is detected.
* Screenshot capture is controlled by a cooldown to prevent excessive duplicate screenshots.


## 🎯 Use Cases

This project can be used for:

* Workplace safety monitoring
* Educational demonstrations
* Real-time computer vision projects
* AI/ML internships
* College projects
* Hackathons

---

### 🚀 Built for Real-Time AI/ML Demonstrations and Hackathons
