# Computer_vision_semester_project
# **Machine Fault Detection System**

## **Project Overview**
This project focuses on developing an **Automatic Machine Fault Detection and Recognition System** using **audio signals and machine learning**. By leveraging **computer vision, deep learning, and signal processing techniques**, the system analyzes machine-generated sounds to detect and classify different types of faults.

## **Table of Contents**
1. [Introduction](#introduction)
2. [Objectives](#objectives)
3. [Technologies Used](#technologies-used)
4. [Dataset](#dataset)
5. [Methodology](#methodology)
6. [Model Development](#model-development)
7. [Results](#results)
8. [Installation & Usage](#installation--usage)
9. [Future Enhancements](#future-enhancements)
10. [Contributors](#contributors)

## **Introduction**
**Machine fault detection** is crucial for predictive maintenance in industries. Traditional methods rely on **manual inspections**, which can be **time-consuming** and **error-prone**. This project aims to **automate the fault detection process** by analyzing machine-generated **audio signals** using **deep learning models**, particularly **Convolutional Neural Networks (CNNs)**.

## **Objectives**
- Develop an **automatic fault detection system** using machine learning.
- Analyze **audio signals** from machines to classify different fault types.
- Utilize **deep learning (CNNs)** and **signal processing** for feature extraction.
- Implement **real-time fault detection** capabilities.

## **Technologies Used**
- **Programming Language:** Python
- **Deep Learning Framework:** TensorFlow, Keras
- **Libraries:**
  - **Librosa** (Audio processing & feature extraction)
  - **Matplotlib & Seaborn** (Data visualization)
  - **NumPy & Pandas** (Data handling)
- **Development Environment:** Google Colab, Jupyter Notebook

## **Dataset**
- The dataset consists of **machine audio recordings** stored in a ZIP file.
- Audio signals are converted into **Mel Spectrograms, MFCCs, and Chroma Features**.
- Data augmentation techniques (noise addition, pitch shifting) are applied to improve model generalization.
- The dataset is split into **training (70%) and testing (30%) sets**.

## **Methodology**
1. **Data Preprocessing:**
   - Extracting ZIP files containing machine audio recordings.
   - Converting audio into **spectrogram images**.
   - Normalizing and resizing spectrograms.
2. **Feature Extraction:**
   - **Mel-Frequency Cepstral Coefficients (MFCCs)** for frequency-based analysis.
   - **Chroma features** for harmonic content.
   - **Mel Spectrograms** for frequency distribution over time.
3. **Model Training & Evaluation:**
   - Train a **CNN model** to recognize fault patterns.
   - Optimize using **Adam optimizer & Binary Cross-Entropy loss function**.
   - Evaluate model performance using **accuracy, F1-score, and confusion matrix**.

## **Model Development**
- The core **deep learning model** is a **CNN-based classifier** with the following layers:
  - **Convolutional layers** (Extract spatial features from spectrograms)
  - **ReLU activation** (Introduces non-linearity for better learning)
  - **Max Pooling layers** (Reduces dimensionality while retaining key features)
  - **Fully connected layers** (Classifies extracted features into different fault categories)
  - **Dropout layers** (Prevents overfitting during training)
 
  - <img width="221" alt="jmd" src="https://github.com/user-attachments/assets/a74ac545-5ed7-4c6f-a102-c247dd176f90" />
  - The class "Looseness" is one of the four fault types being classified in the confusion matrix. It is represented in the third row of the matrix, indicating how well the model predicts it compared to the actual occurrences.

Why is "Looseness" classified?
In the context of fault detection in electrical machines, "Looseness" refers to mechanical faults where machine components (e.g., bolts, bearings, or mountings) are not tightly secured, leading to vibrations and irregular movements. These conditions can generate unique sound patterns or spectrogram features that a CNN-based classifier can detect.

How well is "Looseness" classified?
The model correctly classified 407 instances of "Looseness" (true positives).

It misclassified 9 instances as "Arcing," 9 as "Corona," and 46 as "Tracking."

The majority of the samples were correctly classified, indicating that the model effectively distinguishes "Looseness" from other faults, though there are some misclassifications.


## **Results**
- The **CNN model achieved over 60% accuracy** in detecting machine faults.
- **Key Findings:**
  - Successfully **converted audio signals** into meaningful spectrogram images.
  - Data augmentation significantly improved **model generalization**.
  - Model performed well in distinguishing **different fault types**.

## **Installation & Usage**
### **1. Clone the Repository**
```bash
git clone https://github.com/your-username/machine-fault-detection.git
cd machine-fault-detection
```
### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```
### **3. Run the Model**
```bash
python train_model.py
```
### **4. Test with Custom Audio**
```bash
python predict.py --file_path path/to/audio.wav
```

## **Future Enhancements**
- Fine-tune **pretrained models** (e.g., ResNet, Transformer-based architectures) for better accuracy.
- Implement **real-time audio classification** for live predictions.
- Expand the dataset with more diverse machine sounds.

## **Contributors**
- **Jamshiad CH(12)** (2021-UMDB-000740)
- **Nohman Hameed(14)** (2021-UMDB-000742)
- **Abdul Haq Pirzada(15)** (2021-UMDB-000743)

## **Supervisor**
- **Engr. Ahmed Khwaja**



