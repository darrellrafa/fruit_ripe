# 🔥 Your RexNet Model Integration Guide

Your notebook uses **rexnet_150** with **64x64 input size**. Here's how to integrate it with the web app.

## 📋 What I Found in Your Notebook

- **Model**: `rexnet_150` (timm library)
- **Input Size**: `64x64` pixels  
- **Save Location**: `saved_models/fruits_best_model.pth`
- **Classes**: Defined in your `classes` variable
- **Preprocessing**: ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

## 🚀 Step-by-Step Integration

### Step 1: Add Export Cell to Your Notebook

Open your notebook: `models/f1-0-98-fruits-ripeness-classification-torch.ipynb`

**Add this as a NEW CELL at the very end:**

```python
# ===== EXPORT MODEL FOR WEB APP =====
import torch
import timm
import json
import zipfile
from pathlib import Path

print("🚀 Exporting your rexnet_150 model for web application...")

# Create export directory
export_dir = Path("fruit_model_for_webapp")
export_dir.mkdir(exist_ok=True)

# Recreate and load your trained model
print("📦 Loading your trained model...")
model = timm.create_model(model_name=model_name, pretrained=False, num_classes=len(classes)).to(device)
model.load_state_dict(torch.load(f"{save_dir}/{save_prefix}_best_model.pth", map_location=device))
model.eval()

# Save the complete model
model_path = export_dir / "model.pth"
torch.save(model, model_path)
print(f"✅ Model saved to: {model_path}")

# Create labels.txt
class_names = list(classes.keys())
labels_path = export_dir / "labels.txt"
with open(labels_path, 'w') as f:
    for class_name in class_names:
        f.write(f"{class_name}\n")
print(f"✅ Labels: {class_names}")

# Create config.json with your specific settings
config = {
    "model_name": model_name,
    "labels": class_names,
    "input_size": [64, 64],  # Your specific input size
    "num_classes": len(classes),
    "model_type": "pytorch",
    "architecture": "rexnet_150",
    "preprocessing": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225]
    }
}

with open(export_dir / "config.json", 'w') as f:
    json.dump(config, f, indent=2)

# Test the model
test_model = torch.load(model_path, map_location='cpu')
dummy_input = torch.randn(1, 3, 64, 64)
with torch.no_grad():
    output = test_model(dummy_input)
    print(f"✅ Test successful! Output shape: {output.shape}")

# Create zip package
zip_path = "rexnet_fruit_model.zip"
with zipfile.ZipFile(zip_path, 'w') as zipf:
    for file_path in export_dir.glob("*"):
        zipf.write(file_path, file_path.name)

print(f"📦 Download: {zip_path}")
print("🎯 Ready for web app integration!")
```

### Step 2: Run the Export Cell

Execute the cell you just added. It will:
- Load your trained model from `saved_models/fruits_best_model.pth`
- Export it as `model.pth`
- Create `labels.txt` with your class names
- Create `config.json` with 64x64 input size settings
- Package everything in `rexnet_fruit_model.zip`

### Step 3: Copy to Web App

1. **Download** the `rexnet_fruit_model.zip` from your notebook environment

2. **Create model directory**:
   ```bash
   mkdir models\fruit_rexnet_model
   ```

3. **Extract the zip** to:
   ```
   models/fruit_rexnet_model/
   ├── model.pth          # Your trained rexnet_150 model
   ├── labels.txt         # Your class names
   └── config.json        # 64x64 input size configuration
   ```

### Step 4: Configure Web App

1. **Update .env file**:
   ```env
   MODEL_PATH=models/fruit_rexnet_model
   ```

2. **Restart Flask app**:
   ```bash
   .\venv\Scripts\python app.py
   ```

## ✅ Expected Output

When you restart the app, you should see:
```
[ModelService] Using device: cpu (or cuda)
[ModelService] PyTorch model loaded successfully
[ModelService] Using input size: 64x64
[ModelService] Labels: ['fresh_apple', 'rotten_apple', ...]
```

## 🧪 Testing Your Model

1. **Upload Test**: Go to http://127.0.0.1:5000 and upload a fruit image
2. **Camera Test**: Use the live camera feature
3. **Check Logs**: Monitor Flask console for any errors

## 🔧 What's Already Configured

- ✅ **timm library** installed for rexnet_150 support
- ✅ **64x64 input size** automatically detected from config.json
- ✅ **ImageNet normalization** applied correctly
- ✅ **Class parsing** for fruit type and ripeness detection
- ✅ **GPU/CPU detection** automatic

## 🎯 Your Model is Ready!

The web application will now use your trained rexnet_150 model for:
- File upload analysis
- Live camera analysis  
- Real-time fruit ripeness detection

Your deep learning model is now powering a full web application! 🚀
