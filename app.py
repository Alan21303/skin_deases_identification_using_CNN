from flask import Flask, request, render_template, jsonify
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import io
import base64
from PIL import Image

app = Flask(__name__)

# Load the saved model
model = load_model('model/skin_dis.h5')

# Define class labels
class_labels = {
    0: 'Actinic keratosis',
    1: 'Basal cell carcinoma',
    2: 'Benign keratosis-like lesions',
    3: 'Dermatofibroma',
    4: 'Melanocytic nevi',
    5: 'Melanoma',
    6: 'Vascular lesion'
}

# Function to preprocess the image
def preprocess_image(img, target_size=(28, 28)):
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize to [0, 1] range
    return img_array

# Function to predict the class of the image
def predict_image(img):
    img_array = preprocess_image(img)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_label = class_labels.get(predicted_class, 'Unknown')
    return predicted_class, predicted_label

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        try:
            image_file = Image.open(file.stream)
            
            # Predict the image
            predicted_class, predicted_label = predict_image(image_file)
            
            # Print prediction details to the terminal
            print(f"\nPredicted Class Index: {predicted_class}")
            print(f"\nPredicted Disease Name: {predicted_label}")
            
            # Convert the image to displayable format
            img_buffer = io.BytesIO()
            image_file = image_file.resize((28, 28))  # Resize for display
            image_file.save(img_buffer, format='PNG')
            img_str = base64.b64encode(img_buffer.getvalue()).decode()
            
            # Render result.html with the prediction and image
            return render_template('result.html', prediction=predicted_label, img_data=img_str)
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return jsonify({'error': f'Error during prediction: {str(e)}'})

    return jsonify({'error': 'Invalid request'})

if __name__ == '__main__':
    app.run(debug=True)
