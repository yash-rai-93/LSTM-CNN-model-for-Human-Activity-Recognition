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
<img width="890" height="238" alt="image" src="https://github.com/user-attachments/assets/5941de95-7371-4aa5-8d22-9c0d8103bcc3" />

---
<img width="1100" height="854" alt="image" src="https://github.com/user-attachments/assets/941c50a9-f3b1-4dd9-87ec-bae647d591ce" />

## 📊 Visualization of Accelerometer Data
Add the plot of accelerometer data (X, Y, Z axes):
<img width="907" height="865" alt="image" src="https://github.com/user-attachments/assets/ef4819de-3225-4d5d-86b7-bb913cebf7b3" />


---

## 🧠 Model Architecture

<img width="824" height="166" alt="image" src="https://github.com/user-attachments/assets/090bd676-3ec7-44fc-b823-0605dfc62aa2" />


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
After Training our model gives an accuracy of 98.02% and a loss of 0.58%. The F1 score of training comes out to be 0.96.
<img width="391" height="273" alt="image" src="https://github.com/user-attachments/assets/f939a4a6-d2b3-49d7-9a81-e048fa3120fc" />
=
Now after evaluation of our test data set we get an accuracy of 89.14% and a loss of 46.47%. The F1 score of testing comes out to be 0.89.
<img width="777" height="380" alt="image" src="https://github.com/user-attachments/assets/b55b864c-6538-46a9-b0c1-895411bdd879" />


---

### 🔹 Confusion Matrix
Add confusion matrix image:
<img width="321" height="270" alt="image" src="https://github.com/user-attachments/assets/fb7ad178-e924-4aab-8627-cd3978e8b139" />
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

