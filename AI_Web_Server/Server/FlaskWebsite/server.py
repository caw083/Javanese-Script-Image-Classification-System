import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify, render_template

# Initialize Flask app
app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'  # You can change the folder name as needed
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model
MODEL_PATH = "modelC.h5"  # Replace with your model's path
model = load_model(MODEL_PATH)

# Aksara class names
aksara_classes = [
    'ba', 'ca', 'da', 'dha', 'ga', 'ha', 'ja', 'ka', 'la', 'ma', 'na',
    'nga', 'nya', 'pa', 'ra', 'sa', 'ta', 'tha', 'wa', 'ya'
]

# Preprocessing function
def preprocess_roi(roi, img_width, img_height, channels=1, sharpen=True, denoise=True):
    """
    Preprocess an ROI for model prediction, including optional image enhancement.
    """
    # Optional: Apply sharpening
    if sharpen:
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        roi = cv2.filter2D(roi, -1, kernel)
   
    # Resize the ROI to the model's input dimensions
    roi = cv2.resize(roi, (img_width, img_height))
    
    # If the model expects RGB (3 channels), convert the ROI to 3 channels
    if channels == 3:
        roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)

    # Normalize pixel values to the range [0, 1]    
    # Expand dimensions to match the batch and channel format expected by the model
    roi = np.expand_dims(roi, axis=0)  # Add batch dimension (1, img_width, img_height, channels)

    return roi


# Home route with HTML form
@app.route('/')
def home():
    return render_template('form.html')

# Route to handle image upload and prediction
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return "No image file provided", 400
    
    image = request.files['image']
    if image.filename == '':
        return "No selected file", 400

    # Save the uploaded file
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
    image.save(image_path)

    # Read the uploaded image
    img = cv2.imread(image_path)
    if img is None:
        return "Error reading the image", 400

    # Convert to grayscale and apply threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    ctrs, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours from left to right
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

    # Loop through sorted contours and predict for each ROI
    predictions = []
    for i, ctr in enumerate(sorted_ctrs):
        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)

        # Extract ROI
        roi = thresh[max(0, y-3):y+h+3, max(0, x-3):x+w+3]

        # Preprocess the ROI for prediction
        roi_preprocessed = preprocess_roi(roi, img_width=150, img_height=150, channels=3)

        # Predict the class
        prediction = model.predict(roi_preprocessed)
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        predicted_class_name = aksara_classes[predicted_class_index]

        # Draw bounding box and add the predicted class name
        cv2.rectangle(img, (x-3, y-3), (x + w + 3, y + h + 3), (255, 0, 255), 1)
        cv2.putText(img, predicted_class_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Store the prediction for each contour
        predictions.append({
            "bbox": [x, y, w, h],
            "prediction": predicted_class_name
        })

    # Save the processed image (with predictions)
    processed_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_' + image.filename)
    cv2.imwrite(processed_image_path, img)

    # Show the processed image using OpenCV (this will

    return jsonify({
        "message": "Image uploaded and processed successfully.",
        "predictions": predictions,
        "processed_image_url": processed_image_path
    })

if __name__ == '__main__':
    app.run(debug=True)
