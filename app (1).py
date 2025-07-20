# 1. app.py (Flask Backend)
from flask import Flask, request, render_template, jsonify
import os
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import RandomHeight, RandomWidth, RandomFlip, RandomZoom, RandomRotation
from flask_cors import CORS
import logging

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app) 
logging.basicConfig(level=logging.INFO)

model_path = "my_model/butterfly_model.h5"
try:
    model = load_model(model_path, custom_objects={
        'RandomHeight': RandomHeight,
        'RandomFlip': RandomFlip,
        'RandomZoom': RandomZoom,
        'RandomRotation': RandomRotation,
        'RandomWidth': RandomWidth
    })
    logging.info("✅ Model loaded successfully with custom layers.")
except Exception as e:
    logging.error("❌ Failed to load model", exc_info=True)
    model = None

butterfly_names = {
    0: 'ADONIS', 1: 'AFRICAN GIANT SWALLOWTAIL', 2: 'AMERICAN SNOOT',
    3: 'AN 88', 4: 'APPOLLO', 5: 'ATALA', 6: 'BANDED ORANGE HELICONIAN',
    7: 'BANDED PEACOCK', 8: 'BECKERS WHITE', 9: 'BLACK HAIRSTREAK',
    10: 'BLUE MORPHO', 11: 'BLUE SPOTTED CROW', 12: 'BROWN SIPROETA',
    13: 'CABBAGE WHITE', 14: 'CAIRNS BIRDWING', 15: 'CHECQUERED SKIPPER',
    16: 'CHESTNUT', 17: 'CLEOPATRA', 18: 'CLODIUS PARNASSIAN', 19: 'CLOUDED SULPHUR',
    20: 'COMMON BANDED AWL', 21: 'COMMON WOOD-NYMPH', 22: 'COPPER TAIL', 23: 'CRECENT',
    24: 'CRIMSON PATCH', 25: 'DANAID EGGFLY', 26: 'EASTERN COMA', 27: 'EASTERN DAPPLE WHITE',
    28: 'EASTERN PINE ELFIN', 29: 'ELBOWED PIERROT', 30: 'GOLD BANDED', 31: 'GREAT EGGFLY',
    32: 'GREAT JAY', 33: 'GREEN CELLED CATTLEHEART', 34: 'GREY HAIRSTREAK', 35: 'INDRA SWALLOW',
    36: 'IPHICLUS SISTER', 37: 'JULIA', 38: 'LARGE MARBLE', 39: 'MALACHITE', 40: 'MANGROVE SKIPPER',
    41: 'MESTRA', 42: 'METALMARK', 43: 'MILBERTS TORTOISESHELL', 44: 'MONARCH', 45: 'MOURNING CLOAK',
    46: 'ORANGE OAKLEAF', 47: 'ORANGE TIP', 48: 'ORCHARD SWALLOW', 49: 'PAINTED LADY',
    50: 'PAPER KITE', 51: 'PEACOCK', 52: 'PINE WHITE', 53: 'PIPEVINE SWALLOW', 54: 'POPINJAY',
    55: 'PURPLE HAIRSTREAK', 56: 'PURPLISH COPPER', 57: 'QUESTION MARK', 58: 'RED ADMIRAL',
    59: 'RED CRACKER', 60: 'RED POSTMAN', 61: 'RED SPOTTED PURPLE', 62: 'SCARCE SWALLOW',
    63: 'SILVER SPOT SKIPPER', 64: 'SLEEPY ORANGE', 65: 'SOOTYWING', 66: 'SOUTHERN DOGFACE',
    67: 'STRAITED QUEEN', 68: 'TROPICAL LEAFWING', 69: 'TWO BARRED FLASHER', 70: 'ULYSES',
    71: 'VICEROY', 72: 'WOOD SATYR', 73: 'YELLOW SWALLOW TAIL', 74: 'ZEBRA LONG WING'
}

UPLOAD_FOLDER = os.path.join('static', 'images')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    logging.info("🔁 Received a prediction request")
    if model is None:
        logging.error("🚫 Model is not loaded")
        return jsonify({'error': 'Model not loaded'}), 500

    file = request.files.get('image') or request.files.get('file')
    if not file or file.filename == '':
        logging.error("🚫 No image uploaded or invalid file name")
        return jsonify({'error': 'No image uploaded'}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    logging.info(f"📂 Saving uploaded image to {filepath}")
    file.save(filepath)

    try:
        img = load_img(filepath, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        preds = model.predict(img_array)
        pred_class = np.argmax(preds)
        confidence = float(np.max(preds))

        logging.info(f"✅ Prediction: {butterfly_names.get(pred_class)} ({confidence:.2f})")

        result = {
            'class': butterfly_names.get(pred_class, "Unknown"),
            'confidence': confidence,
            'filename': file.filename
        }
        return jsonify(result)

    except Exception as e:
        logging.exception("❌ Prediction failed")
        return jsonify({'error': 'Prediction failed'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
