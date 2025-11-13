# Human Activity Recognition using CNN-LSTM

## 📌 Introduction
Human Activity Recognition (HAR) is a technique that identifies human actions using Artificial Intelligence (AI) from raw sensor data collected by devices such as smartphones and smartwatches. These devices use sensors like accelerometers, gyroscopes, and magnetometers to measure movement and generate signals based on human activity. HAR has a wide range of applications—healthcare monitoring, fitness tracking, security, gaming, and assisting individuals with disabilities.

There are two broad categories of HAR systems:
- **Fixed Sensors** (installed in environment)
- **Mobile Sensors** (smartphones, wearables)

In this project, we use raw data from **mobile sensors** to classify six activities:
- Downstairs
- Jogging
- Sitting
- Standing
- Upstairs
- Walking

To build an accurate and efficient model, we use a hybrid approach combining **Long Short-Term Memory (LSTM)** networks and **Convolutional Neural Networks (CNN)**. LSTMs capture temporal dependencies in sequential data, while CNNs extract spatial/pattern-based features. This hybrid architecture outperforms basic ML models that depend on manual feature extraction.

---

## 📌 Overview
This project performs **Human Activity Recognition (HAR)** using a hybrid **Convolutional Neural Network (CNN)** and **Long Short-Term Memory (LSTM)** model. The goal is to classify human activities from sequential sensor data or video/keypoint sequences.

The CNN extracts spatial features, while the LSTM captures temporal patterns, making this model ideal for time‑series and sequential human‑motion tasks.

---

## 🎯 Objectives
- Preprocess sensor/video/keypoint dataset for sequential modelling.
- Build a combined **CNN + LSTM deep learning architecture**.
- Train the model to classify different activities.
- Evaluate performance using accuracy, confusion matrix, and loss/accuracy curves.

---

## 📁 Project Structure
```
HUMAN-ACTIVITY-RECOGNITION-LSTM-CNN/
├── .ipynb_checkpoints/
│   └── model-checkpoint.ipynb
├── Dataset/
│   └── WISDM_ar_v1.1/
│       ├── readme.txt
│       ├── WISDM_ar_v1.1_raw_about.txt
│       ├── WISDM_ar_v1.1_raw.txt
│       ├── WISDM_ar_v1.1_trans_about.txt
│       └── WISDM_ar_v1.1_transformed.arff
├── Images/
│   ├── HAR_diag.png
│   └── Model Arch.png
├── LICENSE
├── model.ipynb
└── README.md
```
├── dataset/               # Raw and processed data
├── models/                # Saved trained models
├── notebooks/             # Jupyter notebooks for exploration
├── src/
│   ├── preprocessing.py   # Data loading & preprocessing
│   ├── model.py           # CNN-LSTM architecture
│   ├── train.py           # Training loop
│   ├── evaluate.py        # Model evaluation
│   └── utils.py           # Helper functions
├── results/               # Plots, logs, metrics
└── README.md              # Project documentation
```

---

## 🧠 Working of HAR
Add your image here:

![Working of HAR](Images/HAR_diag.png)

---

## 📊 Visualization of Accelerometer Data
Add the plot of accelerometer data (X, Y, Z axes):

![Accelerometer Visualization](Images/accel_visualization.png)

(Replace `accel_visualization.png` with your actual file name.)

---

## 🧠 Model Architecture
Insert your model architecture diagram here:

![Model Architecture](Images/Model Arch.png)

---
### **1️⃣ CNN Block**
- Extracts spatial features from each frame or timestep.
- Uses Conv2D/Conv1D depending on dataset shape.

### **2️⃣ LSTM Block**
- Captures temporal dependencies in sequential data.
- Processes CNN feature vectors over time.

### **3️⃣ Fully Connected Layer**
- Outputs final activity classification.

---

## ⚙️ Requirements
Install dependencies:
```
pip install -r requirements.txt
```
Key Packages:
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

---

## 🛠️ Usage
### **1. Prepare Dataset**
Update your dataset path inside `preprocessing.py`. Supported formats:
- Time-series sensor data (accelerometer/gyroscope)
- Extracted human pose keypoints
- Frame sequences

Run preprocessing:
```
python src/preprocessing.py
```

### **2. Train the Model**
```
python src/train.py
```
Model will be saved in the `models/` folder.

### **3. Evaluate**
```
python src/evaluate.py
```
Generates:
- Accuracy & Loss graphs
- Confusion matrix
- Evaluation metrics

---

## 📊 Results

### 🔹 Model Accuracy & Loss
Add your training accuracy/loss plot here:

![Accuracy and Loss](Images/accuracy_loss.png)

---

### 🔹 Confusion Matrix
Add confusion matrix image:

![Confusion Matrix](Images/confusion_matrix.png)

--- (Example)
- Training Accuracy: ~95%
- Testing Accuracy: ~92%
- Confusion Matrix indicates strong class separation.

*(Add your project-specific results here.)*

---

## 🔍 Key Features
- Hybrid **CNN-LSTM** architecture
- Modular & clean code design
- Support for different time-series formats
- Visualizations & performance logging
- Model saving & loading included

---

## 🚀 Future Improvements
- Add attention mechanism on LSTM
- Use Transformers for sequence modelling
- Deploy model with Flask/Streamlit
- Improve dataset augmentation

---

## 🤝 Contributing
Pull requests and suggestions are welcome! Feel free to open an issue.

---

## 📚 References
Here are some relevant references and resources for this project:

### Research Papers
- Kun Xia, Jianguang Huang, Hanyu Wang — *LSTM-CNN Architecture for Human Activity Recognition (IEEE)*
- Ordóñez, Francisco Javier & Roggen, Daniel — *Deep Convolutional and LSTM Recurrent Neural Networks for Multimodal Wearable Activity Recognition*
- Hammerla et al. — *Deep, Convolutional, and Recurrent Models for Human Activity Recognition Using Wearables*

### Dataset
- **WISDM Dataset:** https://www.cis.fordham.edu/wisdm/dataset.php

### Tools & Frameworks
- TensorFlow: https://www.tensorflow.org
- Keras API: https://keras.io
- Scikit-learn: https://scikit-learn.org/stable/
- SciPy: https://scipy.org

### Example Implementations / Blogs
- Medium article inspiration: (Add your link here if needed)
- HAR using deep learning examples: https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition
- CNN-LSTM sequence modeling examples: https://keras.io/examples/timeseries/timeseries_classification_from_scratch/

---

## 📜 License
This project is licensed under the **MIT License**.

