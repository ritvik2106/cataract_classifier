# Cataract Classification System

A deep learning-based system for classifying eye images as either normal or having cataracts. This project includes model training using a Jupyter notebook and a Flask API for deployment.

## Project Overview

This system uses a Convolutional Neural Network (CNN) to analyze eye images and classify them into two categories:
- **Normal**: Healthy eye images
- **Cataract**: Eye images showing signs of cataract

## Project Structure

```
cataract_classifier/
├── code/
│   └── train.ipynb          # Model training notebook
├   ── app.py                   # Flask API for deployment
├── requirements.txt         # Python dependencies
|── processed_images
└── README.md               # This file
```

### 2. Run the Training Notebook

```bash
# Start Jupyter notebook
jupyter notebook

# Navigate to code/train.ipynb and run all cells
```

The training notebook will:
- Load and preprocess the dataset
- Split data into training and validation sets
- Build and train the CNN model
- Evaluate model performance
- Save the trained model

### 3. Training Parameters

The model uses the following default parameters:
- **Image Size**: 224x224 pixels
- **Batch Size**: 32
- **Epochs**: 50
- **Learning Rate**: 0.001
- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy

You can modify these parameters in the notebook based on your specific requirements.

## API Deployment

### 1. Start the Flask API

```bash
# Make sure your virtual environment is activated
python3 app.py
```

The API will start running on `http://localhost:5000`

### 2. API Endpoints

#### Predict Cataract
```bash
POST http://localhost:5000/predict
```

**Request Body**: Form data with an image file
- **Content-Type**: `multipart/form-data`
- **Parameter**: `image` (file)

**Response**:
```json
{
    "prediction": "normal", #(or "cataract")
    "confidence": 0.95
}
```

### 3. Testing the API

You can test the API using curl:

```bash
curl -X POST -F "image=@path/to/your/eye/image.jpg" http://localhost:5000/predict
```

Or using Python requests:

```python
import requests

url = "http://localhost:5000/predict"
files = {"image": open("path/to/your/eye/image.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

## File Descriptions

### `code/train.ipynb`
- Jupyter notebook containing the complete training pipeline
- Data preprocessing and augmentation
- Model architecture definition
- Training and evaluation code
- Model saving functionality

### `app.py`
- Flask web application for API deployment
- Image preprocessing for inference
- Model loading and prediction
- Error handling and response formatting


## Environment Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd cataract_classifier
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Jupyter notebook to train the model

### 5. Start the Flask API

```bash
python3 app.py`
```

### 6 Predict a test file

```bash
curl -X POST -F "image=@path/to/your/eye/image.jpg" http://localhost:5000/predict
```