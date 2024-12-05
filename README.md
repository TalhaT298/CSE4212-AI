# Artificial Intelligence Lab Experiments CSE4212

## Experiment List.

### 1. Build a Neural Network Model
- Design and implement a Neural Network model with specified parameters.

### 2. Polynomial Equation Training
- Use the following polynomial equation to train the specified Neural Network:
  
  ```
  3x^3 + 7x^2 − 12x + 2
  ```
- Generate random train and test datasets to evaluate the model.

### 3. Design a Custom DNN for Handwritten Digit Classification
- Train the model using the **MNIST Dataset**.

### 4. Data Augmentation with MNIST Dataset (DNN Model)
- Apply **Data Augmentation** techniques to the MNIST Dataset.
- Compare the results of the augmented dataset with the original dataset for the DNN model.

### 5. Design a Custom CNN for Handwritten Digit Classification

#### Specifications:
1. **CNN Architecture**:
   - Two CNN hidden layers (**Conv2D**) with sizes:
     - 32
     - 64
   - **ReLU Activation** function.
   - **MaxPooling2D** with:
     - Kernel size: (3, 3)
     - Stride: (1, 1)

2. **Flatten Layer**:
   - Convert the feature map into 1D.

3. **Dense Layers**:
   - Simple Dense layer of size 64.
   - Output Dense layer of size 10 with **SoftMax Activation** function.

4. **Dataset**:
   - Use the **MNIST database** for training and testing.

5. **Implementation**:
   - Carefully read the problem specifications and implement the CNN accordingly.

### 6. Data Augmentation with MNIST Dataset (CNN Model)
- Apply **Data Augmentation** techniques to the MNIST Dataset.
- Compare the results of the augmented dataset with the original dataset for the CNN model.

### 7. Design a Custom CNN Classifier Model for MNIST Fashion Dataset
- Design and implement a CNN model for classifying images in the **MNIST Fashion Dataset**.

### 8. Transfer Learning (TL) for Handwritten Digit Classification

#### Specifications:
1. **Base Model**:
   - Use **ResNet50** as the base model.
   - Freeze fully connected (FC) and output layers of ResNet50.

2. **New Layers**:
   - Add a new FC layer with a **Dense layer of size 512**.
   - Add an output Dense layer of size 10 with **SoftMax Activation**.

3. **Weights Initialization**:
   - Initialize weights using **ImageNet**.

4. **Dataset**:
   - Use the **MNIST database** for training and testing.

5. **Data Augmentation**:
   - Apply Data Augmentation to training and test sets of MNIST before training.

6. **Implementation**:
   - Carefully read the problem specifications and implement the TL model accordingly.
