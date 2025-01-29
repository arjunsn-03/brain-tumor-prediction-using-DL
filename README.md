# Brain Tumor Prediction using Deep Learning

## Overview
This project focuses on **brain tumor classification** using **Deep Learning**. It leverages **Convolutional Neural Networks (CNNs)** to classify MRI images into four categories:
- **Glioma**
- **Meningioma**
- **No Tumor**
- **Pituitary Tumor**

The dataset consists of **MRI images** that help in detecting and categorizing brain tumors. The model is trained using **TensorFlow/Keras** and is optimized for high accuracy and robustness.

---

## Dataset
The dataset used in this project is the **Brain Tumor MRI Dataset** from Kaggle.

- **Training images**: `/Training`
- **Testing images**: `/Testing`
- The dataset contains **four classes** of brain tumors and non-tumor images.
- Images are resized to **150x150 pixels** to maintain consistency in the input data.

You can download the dataset from [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset).

---

## Installation & Setup
To run the project locally, follow these steps:

### **1️⃣ Clone the Repository**
```sh
git clone https://github.com/arjunsn-03/brain-tumor-prediction-using-DL.git
cd brain-tumor-prediction-using-DL
```

### **2️⃣ Install Dependencies**
```sh
pip install -r requirements.txt
```

### **3️⃣ Run the Notebook**
You can run the **Jupyter Notebook** or **Google Colab**:
```sh
jupyter notebook
```
Then open `brain-tumour.ipynb` and run the cells.

---

## Model Architecture & Approach
The model is built using **Convolutional Neural Networks (CNNs)**, a powerful deep learning approach for image classification. CNNs work by extracting hierarchical features from images through a series of **convolutional layers**, **activation functions**, **pooling layers**, and **fully connected layers**.

### **1️⃣ Why CNNs?**
CNNs are ideal for image classification because:
- They automatically learn spatial hierarchies of features.
- They are translation invariant, meaning they can detect tumors anywhere in the image.
- They reduce the number of parameters compared to fully connected networks, improving efficiency.

### **2️⃣ Model Architecture**
The implemented CNN follows this architecture:
1. **Input Layer:** Accepts 150x150 pixel images.
2. **Conv2D Layers:** Apply convolution operations to extract features.
3. **ReLU Activation:** Introduces non-linearity for better learning.
4. **MaxPooling:** Downsamples the feature maps, reducing dimensionality.
5. **Dropout Layers:** Prevents overfitting by randomly deactivating neurons during training.
6. **Fully Connected (Dense) Layers:** Flattens extracted features and classifies them into one of four categories.
7. **Softmax Activation:** Converts output into probabilities for classification.

### **3️⃣ Training Process**
- The dataset is split into **80% training** and **20% testing**.
- The model is trained using the **Adam optimizer** and **categorical cross-entropy loss**.
- A **learning rate scheduler** is used to fine-tune the training process.
- **Early Stopping** prevents overfitting by halting training when validation loss stops improving.

---

## Results
- Achieved **high accuracy (above 94%)** on the test dataset.
- Successfully differentiates between different types of brain tumors.
- Model generalizes well across new images due to regularization techniques.

### **Performance Metrics:**
- **Accuracy**: Measures correct classifications.
- **Precision & Recall**: Evaluates model sensitivity to different tumor types.
- **Confusion Matrix**: Provides insights into misclassifications.

---

## Future Improvements
- **Fine-tuning the CNN architecture** for even better accuracy.
- **Data Augmentation** to improve generalization.
- **Transfer Learning** with pre-trained models like **VGG16, ResNet** for performance boost.
- **Hyperparameter tuning** to optimize learning rates and dropout values.
- **Deploying the model** as a web app for real-world usability.

---

