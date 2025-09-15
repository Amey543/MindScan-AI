# MindScan-AI: Brain Tumor Detection 🧠

A deep learning model for classifying brain tumors from MRI images. This project utilizes transfer learning with a VGG16-based architecture to accurately identify different types of brain tumors or determine if no tumor is present.

---

## 📊 Model Performance

The model achieves an overall accuracy of **96%** on the test dataset. The detailed performance metrics for each class are provided below:

| Class         | Precision | Recall | F1-Score |
| :------------ | :-------- | :----- | :------- |
| **Glioma** | 0.95      | 0.93   | 0.94     |
| **Meningioma**| 0.93      | 0.91   | 0.92     |
| **No Tumor** | 0.99      | 1.00   | 0.99     |
| **Pituitary** | 0.95      | 0.99   | 0.97     |

---

## 🛠️ Technology Stack

-   **Machine Learning**: TensorFlow, Keras, Scikit-learn
-   **Core Libraries**: NumPy, Pillow, Matplotlib, Seaborn
-   **Prediction Server**: Flask

---

## 🧠 Model Training

The `model_training_code/` directory contains the Python script used to train the classification model.

### Model Architecture

The model is a Convolutional Neural Network (CNN) built using the **VGG16** architecture with pre-trained ImageNet weights.

-   The base VGG16 model's convolutional layers are frozen, except for the last four, to leverage learned features.
-   A custom classifier head is added on top, consisting of:
    -   A `Flatten` layer
    -   A `Dense` layer with 128 neurons (ReLU activation)
    -   `Dropout` layers for regularization
    -   A final `Dense` output layer with `softmax` activation for multi-class classification.

### Training Process

The training script performs the following steps:
1.  Loads and shuffles the training and testing datasets from local directories.
2.  Applies image augmentation (brightness, contrast) and normalization.
3.  Uses a data generator to feed images to the model in batches.
4.  Compiles the model using the **Adam optimizer** and **sparse categorical cross-entropy** loss function.
5.  Trains the model and evaluates its performance using a classification report, confusion matrix, and ROC curve.
6.  Saves the final trained model as `my_model.keras`.

---

## 🚀 Running the Prediction Server (Flask)

To use the trained model, a simple Flask server is provided in the `backend/` directory.

### Prerequisites

-   Python 3.8+
-   A trained model file named `my_model.keras`.

### Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/MindScan-AI.git](https://github.com/your-username/MindScan-AI.git)
    cd MindScan-AI
    ```

2.  **Navigate to the backend directory:**
    ```bash
    cd backend
    ```

3.  **Create and activate a virtual environment:**
    -   On Windows:
        ```bash
        python -m venv venv
        venv\Scripts\activate
        ```
    -   On macOS/Linux:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```

4.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Place the model file:**
    Ensure your trained model file, `my_model.keras`, is located inside the `backend/` directory.

### Running the Server

1.  **Start the Flask server:**
    ```bash
    python app.py
    ```

2.  **Open your browser:**
    Navigate to **`http://122.0.0.1:5000`** to access the web interface and upload an image for prediction.
