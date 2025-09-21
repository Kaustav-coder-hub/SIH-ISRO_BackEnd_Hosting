from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import base64
import io
from PIL import Image
import numpy as np

app = Flask(__name__)
CORS(app)

# Configuration for ML service
ML_SERVICE_URL = "https://YOUR_USERNAME-sih-isro-ml-service.hf.space/api/predict"  # Replace with your actual URL

@app.route('/')
def home():
    return jsonify({'message': 'SIH-ISRO API Gateway', 'status': 'running'})

@app.route('/api/analyze', methods=['POST'])
def analyze_thermal():
    """
    Main API endpoint - receives thermal image and forwards to ML service
    """
    try:
        # Get uploaded file
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        
        # Basic validation and preprocessing with PIL (lightweight)
        image = Image.open(file.stream)
        
        # Convert to base64 for ML service
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode()
        
        # Forward to ML service
        ml_response = requests.post(
            ML_SERVICE_URL,
            json={
                'image': img_b64,
                'parameters': request.form.get('parameters', '{}')
            },
            timeout=30
        )
        
        if ml_response.status_code == 200:
            return jsonify(ml_response.json())
        else:
            return jsonify({'error': 'ML service unavailable'}), 503
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/status')
def status():
    """Health check endpoint"""
    return jsonify({'api': 'healthy', 'ml_service': 'checking...'})

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
