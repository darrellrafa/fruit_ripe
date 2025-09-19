# 🔥 PyTorch Model Integration - Quick Start

Your web application now supports PyTorch models! Here's how to integrate your existing notebook.

## ⚡ Quick Steps

### 1. Export from Your Notebook
Open your existing notebook: `f1-0-98-fruits-ripeness-classification-torch.ipynb`

Add these cells at the end (after training):

```python
# Export your trained model
import torch
from pathlib import Path
import json

# Create export directory
export_dir = Path("fruit_model_export")
export_dir.mkdir(exist_ok=True)

# Save model (replace 'model' with your actual model variable)
torch.save(model, export_dir / "model.pth")

# Create labels.txt with your class names
class_names = ["fresh_apple", "rotten_apple", "fresh_banana", "rotten_banana"]  # Update this!
with open(export_dir / "labels.txt", 'w') as f:
    for name in class_names:
        f.write(f"{name}\n")

print("✅ Model exported successfully!")
```

### 2. Copy to Web App
1. Download the exported files from your notebook
2. Create folder: `models/fruit_model_pytorch/`
3. Place files:
   ```
   models/fruit_model_pytorch/
   ├── model.pth
   └── labels.txt
   ```

### 3. Configure & Run
1. Update `.env`:
   ```
   MODEL_PATH=models/fruit_model_pytorch
   ```

2. Restart Flask app:
   ```bash
   .\venv\Scripts\python app.py
   ```

## ✅ What's Already Done

- ✅ PyTorch support added to model service
- ✅ PyTorch installed in your environment  
- ✅ Automatic model type detection (PyTorch vs TensorFlow)
- ✅ Compatible with your existing web interface
- ✅ Works with both file upload and live camera

## 🎯 Expected Output

When you restart the app, you should see:
```
[ModelService] PyTorch model loaded successfully
[ModelService] Labels: ['fresh_apple', 'rotten_apple', ...]
```

## 📋 Detailed Guides

- `integrate_pytorch_model.md` - Complete integration guide
- `export_your_model.py` - Copy-paste code for your notebook

## 🔧 Troubleshooting

**Model not loading?**
- Check that `model.pth` and `labels.txt` exist in the model directory
- Verify MODEL_PATH in `.env` points to the correct folder
- Check Flask console for error messages

**Wrong predictions?**
- Ensure labels.txt matches your training class order
- Verify image preprocessing matches your training setup

Your PyTorch model is ready to power the web application! 🚀
