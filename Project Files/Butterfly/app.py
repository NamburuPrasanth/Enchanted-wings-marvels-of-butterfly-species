# Butterfly Species Classification Flask Web Application
# This application classifies butterfly images using a pre-trained VGG16 model.
#
# To run this application:
# 1. Save this code as 'app.py'
# 2. Ensure your trained model file 'best_vgg16_model.h5' is in the same directory
# 3. Create a 'templates' folder with 'index.html'
# 4. Create a 'static' folder with an 'uploads' subfolder
# 5. Install required libraries: pip install Flask tensorflow numpy Pillow werkzeug
# 6. Run: python app.py
# 7. Access at: http://127.0.0.1:5000

# --- Required Libraries ---
import os
import json
from flask import Flask, render_template, request, url_for, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from PIL import Image
from werkzeug.utils import secure_filename

# --- Global Configurations ---
UPLOAD_FOLDER = os.path.join('static', 'uploads')
IMG_TARGET_SIZE = (224, 224)

# Define butterfly species class labels
# IMPORTANT: Update this list with your actual 75 butterfly species in the correct order
# This should match the order from your train_generator.class_indices
BUTTERFLY_SPECIES = [
    'ADONIS', 'AFRICAN GIANT SWALLOWTAIL', 'AMERICAN SNOUT', 'AN 88', 'APPOLLO',
    'ATALA', 'ATLAS MOTH', 'BANDED ORANGE HELICONIAN', 'BANDED PEACOCK', 'BANDED TIGER MOTH',
    'BLUE MORPHO', 'BLUE SPOTTED CROW', 'BROOKES BIRDWING', 'BROWN SIPROETA',
    'CABBAGE WHITE', 'CAIRNS BIRDWING', 'CHECKERSPOT', 'CHESTNUT', 'CLEOPATRA',
    'CLODIUS PARNASSIAN', 'CLOUDED SULPHUR', 'COPPER TAIL', 'CRECENT',
    'CRIMSON PATCH', 'DANAUS CHRYSIPPUS', 'EASTERN COMA', 'EASTERN DAPPLE WHITE',
    'EASTERN PINE ELFIN', 'ELBOWED PIERROT', 'GOLD BANDED', 'GREAT EGGFLY',
    'GREAT JAY', 'GREEN CELLED CATTLEHEART', 'GREY HAIRSTREAK', 'INDRA SWALLOWTAIL',
    'ITHOMIA HERALDICA', 'JULIA', 'LARGE MARBLE', 'LEMON PANSY', 'LIME BUTTERFLY',
    'LONG TAILED SKIPPER', 'LYDIUS', 'MALACHITE', 'MANGROVE SKIPPER', 'MESTRA',
    'METALMARK', 'MILBERTS TORTOISESHELL', 'MONARCH', 'MOURNING CLOAK',
    'ORANGE OAKLEAF', 'ORANGE TIP', 'ORCHARD SWALLOW', 'PAINTED LADY',
    'PAPER KITE', 'PEACOCK', 'PEARL CRESCENT', 'PINE WHITE', 'PIPEVINE SWALLOW',
    'POSTMAN', 'PURPLE HAIRSTREAK', 'QUESTION MARK', 'RED ADMIRAL', 'RED CRACKER',
    'RED POSTMAN', 'RED SPOTTED PURPLE', 'ROCKY MOUNTAIN PARNASSIAN', 'RUBY SPOTTED SWALLOWTAIL',
    'RUSTY TIPPED PAGE', 'SARA LONGWING', 'SHORT TAILED SKIPPER', 'SILVER SPOT SKIPPER',
    'SIXSPOT BURNET', 'SLEEPY ORANGE', 'SOOTYWING', 'SOUTHERN DOGFACE', 'STRAITED QUEEN',
    'TROPICAL LEAFWING', 'TWO BARRED FLASHER', 'ULYSES', 'VICEROY', 'WOOD SATYR',
    'YELLOW SWALLOW TAIL', 'ZEBRA LONG WING'
]

# Create labels mapping
CLASS_LABELS = {i: species for i, species in enumerate(BUTTERFLY_SPECIES)}

# --- Flask Application Initialization ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])
    print(f"Created upload directory: {app.config['UPLOAD_FOLDER']}")

# --- Load the Pre-trained Model ---
model = None
try:
    model = tf.keras.models.load_model('best_vgg16_model.h5')
    print("Butterfly classification model loaded successfully.")
    print(f"Model expects input shape: {model.input_shape}")
    print(f"Model output classes: {len(CLASS_LABELS)}")
except Exception as e:
    print(f"ERROR: Could not load the model 'best_vgg16_model.h5'.")
    print(f"Please ensure the model file is in the same directory as 'app.py'.")
    print(f"Detailed error: {e}")

# --- Utility Functions ---
def get_top_predictions(predictions, top_k=5):
    """Get top-k predictions with confidence scores"""
    top_indices = np.argsort(predictions)[::-1][:top_k]
    top_predictions = []
    
    for i, idx in enumerate(top_indices):
        top_predictions.append({
            'rank': i + 1,
            'species': CLASS_LABELS.get(idx, 'Unknown Species'),
            'confidence': float(predictions[idx]) * 100,
            'probability': float(predictions[idx])
        })
    
    return top_predictions

def format_species_name(species_name):
    """Format species name for better display"""
    return species_name.replace('_', ' ').title()

# --- Flask Routes ---
@app.route('/')
def index_page():
    """
    Serves the single-page application (index.html).
    All functionality is contained within this file with JavaScript handling.
    """
    return render_template("index.html", species_count=len(CLASS_LABELS))

@app.route('/predict_api', methods=['POST'])
def predict_butterfly_api():
    """
    Handles butterfly image uploads and returns JSON prediction results.
    This endpoint is called via AJAX from the frontend.
    """
    # Check if file is present in request
    if 'butterfly_image' not in request.files:
        return jsonify({"error": "No file uploaded. Please select a butterfly image."}), 400

    file = request.files['butterfly_image']
    if file.filename == '':
        return jsonify({"error": "No file selected. Please choose an image to upload."}), 400

    if file:
        # Secure filename and save
        img_filename = file.filename
        secure_img_filename = secure_filename(img_filename)
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_img_filename)

        try:
            file.save(img_path)
            print(f"Butterfly image saved: {img_path}")
        except Exception as e:
            print(f"Error saving image: {e}")
            return jsonify({"error": f"Could not save the uploaded image: {e}"}), 500

        # Check if model is loaded
        if model is None:
            return jsonify({"error": "Classification model not loaded. Please check server logs."}), 500

        try:
            # Preprocess the image
            img = load_img(img_path, target_size=IMG_TARGET_SIZE)
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            img_array = preprocess_input(img_array)  # VGG16 preprocessing

            # Make prediction
            predictions = model.predict(img_array, verbose=0)
            predicted_class_index = np.argmax(predictions[0])
            predicted_species = CLASS_LABELS.get(predicted_class_index, 'Unknown Species')
            confidence = float(np.max(predictions[0])) * 100

            # Get top predictions
            top_predictions = get_top_predictions(predictions[0], top_k=5)

            print(f"Prediction for '{secure_img_filename}': {predicted_species} ({confidence:.2f}%)")

            # Prepare response
            response_data = {
                "success": True,
                "predicted_species": format_species_name(predicted_species),
                "confidence": round(confidence, 2),
                "top_predictions": [
                    {
                        "rank": pred["rank"],
                        "species": format_species_name(pred["species"]),
                        "confidence": round(pred["confidence"], 2)
                    }
                    for pred in top_predictions
                ],
                "uploaded_image_url": url_for('static', filename=os.path.join('uploads', secure_img_filename)),
                "total_species": len(CLASS_LABELS)
            }

            return jsonify(response_data), 200

        except Exception as e:
            print(f"Error during prediction: {e}")
            return jsonify({"error": f"Error processing image: {str(e)}. Please try again with a different image."}), 500

    return jsonify({"error": "An unexpected error occurred."}), 500

@app.route('/get_species_info')
def get_species_info():
    """
    Returns information about all supported butterfly species.
    """
    species_info = {
        "total_species": len(CLASS_LABELS),
        "species_list": [format_species_name(species) for species in BUTTERFLY_SPECIES],
        "model_info": {
            "architecture": "VGG16 Transfer Learning",
            "input_size": f"{IMG_TARGET_SIZE[0]}x{IMG_TARGET_SIZE[1]}",
            "model_loaded": model is not None
        }
    }
    return jsonify(species_info)

@app.route('/health')
def health_check():
    """
    Simple health check endpoint.
    """
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "upload_folder_exists": os.path.exists(app.config['UPLOAD_FOLDER']),
        "supported_species": len(CLASS_LABELS)
    })

# --- Error Handlers ---
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(413)
def too_large(error):
    return jsonify({"error": "File too large. Please upload a smaller image."}), 413

# --- Main Entry Point ---
if __name__ == '__main__':
    print("="*60)
    print("🦋 BUTTERFLY SPECIES CLASSIFICATION WEB APP 🦋")
    print("="*60)
    print(f"📁 Upload folder: {UPLOAD_FOLDER}")
    print(f"🔢 Supported species: {len(CLASS_LABELS)}")
    print(f"🤖 Model loaded: {'✅ Yes' if model else '❌ No'}")
    print(f"📏 Expected image size: {IMG_TARGET_SIZE}")
    print("="*60)
    print("🚀 Starting Flask development server...")
    print("🌐 Access the app at: http://127.0.0.1:5000")
    print("="*60)
    
    # Configure Flask app
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
    
    # Run the application
    app.run(debug=True, host='127.0.0.1', port=5000)