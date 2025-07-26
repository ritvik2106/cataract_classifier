from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image

# Initialize app
app = Flask(__name__)

# Load trained model
MODEL_PATH = './eyes_model.keras'
model = load_model(MODEL_PATH)

# Define class labels
# index 0 = cataract, 1 = normal
CLASS_LABELS = ['cataract', 'normal']

# Image preprocessing function (adjust according to your model input shape)
def preprocess_image(img_path, target_size=(224, 224)):
    '''
    This function is used to preprocess the image.
    It takes an image path and a target size and returns a preprocessed image.
    '''
    img = image.load_img(img_path, target_size=(500, 800))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    '''
    This function is used to predict the class of the image.
    It takes a POST request with a file parameter and returns the predicted class and confidence.
    '''
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    # Save temporarily
    file_path = os.path.join('temp', file.filename)
    os.makedirs('temp', exist_ok=True)
    file.save(file_path)

    try:
        # Preprocess and predict
        img = preprocess_image(file_path)
        preds = model.predict(img)[0]
        predicted_class = CLASS_LABELS[np.argmax(preds)]
        confidence = float(np.max(preds))

        # Clean temp file
        os.remove(file_path)

        return jsonify({
            'prediction': predicted_class,
            'confidence': 1-round(confidence, 3)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True)



# curl -X POST -F "file=@/home/processed_images/test/cataract/image_265.png" http://localhost:5000/predict
