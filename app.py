from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import os
from werkzeug.utils import secure_filename
from datetime import datetime
from models.model_service import FruitRipenessModel
from dotenv import load_dotenv
import base64

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # default; will be overridden by SECRET_KEY if present

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize model service (set MODEL_PATH env var if you have a trained model)
load_dotenv()
MODEL_PATH = os.getenv('MODEL_PATH')
secret_from_env = os.getenv('SECRET_KEY')
if secret_from_env:
    app.secret_key = secret_from_env
model_service = FruitRipenessModel(model_path=MODEL_PATH)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        # Save the uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Call the model service to get prediction
        try:
            pred = model_service.predict(filepath)
            result = {
                'status': 'success',
                'filename': filename,
                'prediction': pred.get('prediction', 'not sure'),
                'confidence': float(pred.get('confidence', 0.0)),
                'fruit_type': pred.get('fruit_type', 'fruit')
            }
            return jsonify(result)
        except Exception as e:
            # Log the error in real apps; here we return a safe message
            return jsonify({'status': 'error', 'message': 'Failed to analyze image.'}), 500
    
    flash('Invalid file type. Please upload an image file (PNG, JPG, JPEG, GIF).')
    return redirect(url_for('index'))

@app.route('/save_frame', methods=['POST'])
def save_frame():
    """Save camera frame to dataset for training"""
    data = request.get_json(silent=True)
    if not data or 'image' not in data or 'class_name' not in data:
        return jsonify({'status': 'error', 'message': 'Missing image or class_name'}), 400

    data_url = data['image']
    class_name = data['class_name']
    split = data.get('split', 'train')  # train, val, or test
    
    if not isinstance(data_url, str) or ',' not in data_url:
        return jsonify({'status': 'error', 'message': 'Invalid image format'}), 400

    # Validate class name (alphanumeric and underscore only)
    if not class_name.replace('_', '').replace('-', '').isalnum():
        return jsonify({'status': 'error', 'message': 'Invalid class name'}), 400

    header, b64data = data_url.split(',', 1)
    ext = 'jpg'
    if 'image/png' in header:
        ext = 'png'

    try:
        img_bytes = base64.b64decode(b64data)
    except Exception:
        return jsonify({'status': 'error', 'message': 'Failed to decode image'}), 400

    # Create class directory if it doesn't exist
    class_dir = os.path.join('data', 'images', split, class_name)
    os.makedirs(class_dir, exist_ok=True)

    # Save with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{timestamp}.{ext}"
    filepath = os.path.join(class_dir, filename)
    
    try:
        with open(filepath, 'wb') as f:
            f.write(img_bytes)
        return jsonify({
            'status': 'success', 
            'message': f'Saved to {class_name}',
            'filepath': filepath,
            'class_name': class_name,
            'split': split
        })
    except Exception:
        return jsonify({'status': 'error', 'message': 'Failed to save image'}), 500

@app.route('/analyze_frame', methods=['POST'])
def analyze_frame():
    data = request.get_json(silent=True)
    if not data or 'image' not in data:
        return jsonify({'status': 'error', 'message': 'No image data provided'}), 400

    data_url = data['image']
    if not isinstance(data_url, str) or ',' not in data_url:
        return jsonify({'status': 'error', 'message': 'Invalid image format'}), 400

    header, b64data = data_url.split(',', 1)
    # Determine extension from header if possible
    ext = 'jpg'
    if 'image/png' in header:
        ext = 'png'
    elif 'image/jpeg' in header or 'image/jpg' in header:
        ext = 'jpg'

    try:
        img_bytes = base64.b64decode(b64data)
    except Exception:
        return jsonify({'status': 'error', 'message': 'Failed to decode image data'}), 400

    # Save to uploads folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{timestamp}_camera.{ext}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        with open(filepath, 'wb') as f:
            f.write(img_bytes)
    except Exception:
        return jsonify({'status': 'error', 'message': 'Failed to save image'}), 500

    # Predict
    try:
        pred = model_service.predict(filepath)
        result = {
            'status': 'success',
            'filename': filename,
            'prediction': pred.get('prediction', 'not sure'),
            'confidence': float(pred.get('confidence', 0.0)),
            'fruit_type': pred.get('fruit_type', 'fruit')
        }
        return jsonify(result)
    except Exception:
        return jsonify({'status': 'error', 'message': 'Failed to analyze frame'}), 500

if __name__ == '__main__':
    app.run(debug=True)
