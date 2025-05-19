# Waste Material Segregation using CNNs

This project implements a Convolutional Neural Network (CNN) model to automatically classify waste materials into categories such as **Plastic, Glass, Metal, Paper, Cardboard, Food Waste**, and **Other**. It is aimed at improving recycling efficiency and promoting sustainable waste management.

## 📂 Project Structure
waste_segg_final/
├── data/ # Dataset (ZIP file or extracted folders)
├── notebooks/ # Colab
├── models/ # Saved model files
├── utils/ # Custom utility scripts (if any)
└── README.md # Project documentation


---

## Objectives

- Build a CNN-based classifier to categorize waste images.
- Improve classification accuracy using data augmentation and regularization techniques.
- Analyze performance through evaluation metrics and confusion matrix.


## Technologies Used

- Python 3.x
- TensorFlow / Keras
- NumPy, Pandas, Matplotlib, Seaborn
- scikit-learn (for evaluation metrics)
- Google Colab (for training and experimentation)


##  Model Architecture

- 3 Convolutional layers (with ReLU + MaxPooling)
- Batch Normalization
- Dropout layers to reduce overfitting
- Fully connected Dense layer
- Output layer with Softmax (7 classes)

---

## Data Preprocessing

- Image resizing to **128×128**
- One-hot encoding of target labels
- Train/Validation split (80/20)
- Real-time data augmentation using `ImageDataGenerator`

---

## Results

- **Best Validation Accuracy:** ~44.9%
- **Key Insight:** Class imbalance affects model performance — *Plastic* and *Food Waste* have high recall, while *Metal* and *Glass* underperform.

---

##  Evaluation

- Accuracy, Loss (train/val)
- Classification Report
- Confusion Matrix

---

## How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/nitishnarayanan002/waste_segg_final.git
2. Open the notebook in Google Colab or Jupyter.

3. Run through the steps:

-- Upload and unzip the dataset(File name - Data.zip) 

-- Load, preprocess, and augment images

--- Train the CNN model

--- Evaluate and visualize results
   


