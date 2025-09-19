# Fruit Ripeness Analyzer

A web application that analyzes images of fruits to determine their ripeness level using deep learning.

## Features

- Upload images of fruits through a user-friendly interface
- Real-time image preview
- Ripeness analysis with confidence scores
- Responsive design that works on desktop and mobile devices
- Support for multiple fruit types (extendable)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd fruit-ripeness-analyzer
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   # On Windows PowerShell
   venv\Scripts\Activate.ps1
   # On Windows CMD
   venv\Scripts\activate.bat
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables:
   - Copy `.env.example` to `.env`
   - Set `SECRET_KEY` to any secure random string
   - If you have a trained TensorFlow model, set `MODEL_PATH` to its directory or file
   
   Example `.env`:
   ```env
   SECRET_KEY=please-change-me-to-a-random-string
   MODEL_PATH=models/my_fruit_model/   # or models/my_model.h5
   ```

## Usage

1. Run the Flask application:
   ```bash
   python app.py
   ```

2. Open your web browser and navigate to:
   ```
   http://localhost:5000
   ```

3. Upload an image of a fruit to analyze its ripeness.

## Project Structure

- `app.py` - Main Flask application
- `templates/` - HTML templates
  - `index.html` - Main web interface
- `static/` - Static files (CSS, JS, uploaded images)
  - `uploads/` - Directory for storing uploaded images
- `models/` - Directory for storing trained models (to be added)
- `requirements.txt` - Python dependencies

## Dataset Collection & Model Training

### Option A: Collect Dataset Using the Web App

1. Start the Flask app and open http://127.0.0.1:5000
2. Go to "Live Camera Analyze" section
3. Click "Start Camera" and allow camera permission
4. Click "Dataset Mode" to show the dataset collection UI
5. Select a class (e.g., "Banana - Ripe") or create custom classes
6. Point camera at fruit and click "Save Frame" to collect images
7. Collect 50-200+ images per class for good results

The images are automatically organized in:
```
data/images/train/[class_name]/timestamp.jpg
```

### Option B: Manual Dataset Organization

Create this folder structure and add your images:
```
data/images/
├── train/
│   ├── banana_ripe/
│   ├── banana_unripe/
│   ├── apple_ripe/
│   └── apple_unripe/
├── val/ (optional)
└── test/ (optional)
```

### Training Your Model

#### Method 1: Local Training (requires Python 3.11 + TensorFlow)

1. Create Python 3.11 environment:
   ```bash
   python3.11 -m venv venv311
   venv311\Scripts\activate
   pip install tensorflow==2.17.0 matplotlib pillow numpy
   ```

2. Run the training script:
   ```bash
   python train_model.py
   ```

3. The script will:
   - Load images from `data/images/train/`
   - Train using transfer learning (MobileNetV2)
   - Save model to `models/fruit_model_[timestamp]/`
   - Update `.env` with the new model path

#### Method 2: Google Colab Training (recommended)

1. Upload `Fruit_Ripeness_Training.ipynb` to Google Colab
2. Upload your dataset to Google Drive
3. Update the `DATA_PATH` in the notebook
4. Run all cells to train the model
5. Download the trained model zip file
6. Extract to your project's `models/` directory

### Using Your Trained Model

1. Set `MODEL_PATH` in `.env`:
   ```env
   MODEL_PATH=models/fruit_model_20241218_143022/
   ```

2. Restart the Flask app:
   ```bash
   .\venv\Scripts\python app.py
   ```

3. The app will automatically load your model and use it for both file uploads and live camera analysis.

### Model Integration Details

- **Input**: 224x224 RGB images, normalized to [0,1]
- **Output**: Softmax probabilities for each class
- **Labels**: Automatically loaded from `labels.txt` in the model directory
- **Fallback**: If model loading fails, the app uses a color-based heuristic

Example `labels.txt`:
```
banana_ripe
banana_unripe
apple_ripe
apple_unripe
```

The model service parses class names to extract fruit type and ripeness:
- `banana_ripe` → fruit_type: "banana", prediction: "ripe"
- `apple_unripe` → fruit_type: "apple", prediction: "not ripe"

## Testing

To run the tests, use the following command:
```bash
python -m unittest discover -s tests
```
Make sure to write unit tests for any new functionality you add.

## Team Members

- [Your Name]
- [Team Member 2]
- [Team Member 3]

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
