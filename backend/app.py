from flask import Flask, render_template, request, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import uuid # To generate unique filenames and avoid conflicts

# --- CHANGE 1: Tell Flask where the 'templates' folder is ---
# Since index.html is in ../frontend, we set the template_folder path.
app = Flask(__name__, template_folder='../frontend')

# Load the trained model
model = load_model('my_model.keras')

# Class labels
class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary'] # Make sure this order matches your model's training

# Define a folder to store uploaded images temporarily
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Helper function to predict tumor type
def predict_tumor(image_path):
    IMAGE_SIZE = 128
    img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence_score = np.max(predictions, axis=1)[0]

    # --- CHANGE 2: Simplified result logic ---
    predicted_label = class_labels[predicted_class_index]
    if predicted_label == 'notumor':
        return "No Tumor Detected", confidence_score
    else:
        # Capitalize the tumor type for better display
        return f"Tumor Detected: {predicted_label.capitalize()}", confidence_score

# Route for the main page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error="No file part")
            
        file = request.files['file']
        
        if file.filename == '':
            return render_template('index.html', error="No selected file")

        if file:
            # --- CHANGE 3: Use a unique filename to prevent browser caching issues ---
            filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
            file_location = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_location)

            # Predict the tumor
            result, confidence = predict_tumor(file_location)
            
            # --- CHANGE 4: Generate a URL for the image and pass it to the template ---
            # We no longer delete the file here, so the frontend can display it.
            image_url = url_for('get_uploaded_file', filename=filename)

            return render_template(
                'index.html',
                result=result,
                confidence=f"{confidence*100:.2f}", # Pass confidence as a formatted number
                file_path=image_url
            )

    return render_template('index.html', result=None, file_path=None)

# Route to serve the uploaded files from the 'uploads' directory
@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    # Use send_from_directory for security
    return Flask.send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    # Use debug=True for local development to see errors
    app.run(host="0.0.0.0", port=5000, debug=True)