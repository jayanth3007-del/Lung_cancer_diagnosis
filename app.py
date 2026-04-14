from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import base64
import numpy as np
from PIL import Image
from datetime import datetime
import io

app = Flask(__name__)
CORS(app)

# Config
UPLOAD_FOLDER = 'static/uploads'
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create folders
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ── Symptom-based risk model ──────────────────────────────────────────────────
# NOTE: This is a rule-based scoring model, not a real trained AI.
# To use a real ML model, replace this class with your model loader.

class SymptomModel:
    def predict(self, symptoms, age, gender, duration):
        s = (symptoms or '').lower()
        risk = 8

        # Symptom scoring
        if 'cough' in s:        risk += 20
        if 'blood' in s:        risk += 26
        if 'pain' in s:         risk += 15
        if 'breath' in s:       risk += 18
        if 'weight' in s:       risk += 12
        if 'fatigue' in s:      risk += 10
        if 'hoarse' in s:       risk += 13
        if 'wheez' in s:        risk += 11
        if 'chest' in s:        risk += 14
        if 'swallow' in s:      risk += 9

        # Age scoring
        if age > 55: risk += 12
        if age > 65: risk += 10

        risk = min(risk, 92)

        if risk < 30:
            risk_level = 'Low'
        elif risk < 58:
            risk_level = 'Medium'
        else:
            risk_level = 'High'

        benign = max(5, 88 - risk)

        return {
            'risk_score': risk,
            'risk_level': risk_level,
            'probabilities': {
                'adenocarcinoma': round(risk * 0.44),
                'squamous_cell':  round(risk * 0.30),
                'large_cell':     round(risk * 0.26),
                'benign':         benign,
            }
        }


# ── Image-based CNN model (still simulated — replace with real model) ─────────
class LungCancerModel:
    def predict(self, image_array):
        features = np.mean(image_array)
        # TODO: Replace this block with a real model, e.g.:
        # result = real_model.predict(image_array)
        probability = float(1 / (1 + np.exp(-features * 2)))
        diagnosis = "MALIGNANT" if probability > 0.5 else "BENIGN"
        confidence = 87  # Fixed value — real model will return real confidence

        return {
            "diagnosis": diagnosis,
            "cancer_probability": round(probability, 3),
            "confidence": confidence,
            "features_detected": 12
        }


symptom_model = SymptomModel()
image_model = LungCancerModel()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def home():
    return send_from_directory('static', 'index.html')


# NEW ROUTE — This is what the frontend was calling but didn't exist
@app.route('/predict_symptoms', methods=['POST'])
def predict_symptoms():
    try:
        data = request.get_json()
        symptoms = data.get('symptoms', '')
        age      = int(data.get('age', 0))
        gender   = data.get('gender', '')
        duration = data.get('duration', '')

        result = symptom_model.predict(symptoms, age, gender, duration)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# Existing route — image-based prediction
@app.route('/api/predict', methods=['POST'])
def predict_image():
    try:
        data = request.get_json()
        image_b64 = data['image'].split(',')[1]

        image_data = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0

        result = image_model.predict(image_array)
        result['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"xray_{timestamp}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(filepath, 'JPEG', quality=85)

        return jsonify({
            "success": True,
            "result": result,
            "image_url": f"/static/uploads/{filename}"
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
