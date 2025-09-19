# Integrating Your PyTorch Model

This guide shows how to export your trained PyTorch model from your existing notebook and integrate it with the web application.

## Step 1: Export Your Model from Your Notebook

Add these cells to the end of your existing notebook (`f1-0-98-fruits-ripeness-classification-torch.ipynb`):

### Cell 1: Save the Model
```python
import torch
import json
from pathlib import Path

# Create model directory
model_dir = Path("fruit_model_pytorch")
model_dir.mkdir(exist_ok=True)

# Save the trained model
model_path = model_dir / "model.pth"
torch.save(model, model_path)  # Replace 'model' with your actual model variable name
print(f"Model saved to: {model_path}")

# If you have a separate state dict, use this instead:
# torch.save(model.state_dict(), model_path)
```

### Cell 2: Create Labels File
```python
# Create labels.txt - update this list with your actual class names
class_names = [
    "fresh_apple", 
    "fresh_banana", 
    "fresh_orange",
    "rotten_apple", 
    "rotten_banana", 
    "rotten_orange"
]  # Replace with your actual class names

# Save labels
labels_path = model_dir / "labels.txt"
with open(labels_path, 'w') as f:
    for class_name in class_names:
        f.write(f"{class_name}\n")
print(f"Labels saved to: {labels_path}")
```

### Cell 3: Create Config File (Optional)
```python
# Create config.json with model metadata
config = {
    "labels": class_names,
    "input_size": [224, 224],  # Update with your model's input size
    "num_classes": len(class_names),
    "model_type": "pytorch",
    "preprocessing": {
        "mean": [0.485, 0.456, 0.406],  # ImageNet defaults
        "std": [0.229, 0.224, 0.225]
    }
}

config_path = model_dir / "config.json"
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)
print(f"Config saved to: {config_path}")
```

### Cell 4: Test the Saved Model
```python
# Test loading the saved model
test_model = torch.load(model_path, map_location='cpu')
test_model.eval()
print("✅ Model loads successfully!")
print(f"Model type: {type(test_model)}")
```

### Cell 5: Create Download Package
```python
import zipfile

# Create zip file for easy download
zip_path = "fruit_model_pytorch.zip"
with zipfile.ZipFile(zip_path, 'w') as zipf:
    for file_path in model_dir.glob("*"):
        zipf.write(file_path, file_path.name)

print(f"📦 Model package created: {zip_path}")
print("Download this file and extract it to your web app's models/ directory")
```

## Step 2: Integrate with Web Application

1. **Download and Extract**: Download the zip file from your notebook and extract it to:
   ```
   c:\Users\PC\CascadeProjects\windsurf-project\models\fruit_model_pytorch\
   ```

2. **Install PyTorch**: Update your requirements and install PyTorch:
   ```bash
   .\venv\Scripts\python -m pip install torch torchvision
   ```

3. **Update .env**: Set the model path in your `.env` file:
   ```env
   MODEL_PATH=models/fruit_model_pytorch
   ```

4. **Restart the App**:
   ```bash
   .\venv\Scripts\python app.py
   ```

## Step 3: Customize for Your Model (if needed)

If your model has specific requirements, you can customize the model service:

### Custom Preprocessing
Edit `models/model_service.py` in the `_load_pytorch_model` method:

```python
# Update the transform to match your training preprocessing
self.transform = transforms.Compose([
    transforms.Resize((your_input_size, your_input_size)),  # e.g., (224, 224)
    transforms.ToTensor(),
    transforms.Normalize(mean=[your_mean], std=[your_std])  # Use your training values
])
```

### Custom Model Loading
If you saved only the state dict, modify the loading code:

```python
# In _load_pytorch_model method, replace:
self.model = torch.load(model_file, map_location=self.device)

# With:
from your_model_definition import YourModelClass  # Import your model class
self.model = YourModelClass(num_classes=len(self.labels))
self.model.load_state_dict(torch.load(model_file, map_location=self.device))
```

## Expected Directory Structure

After integration, your model directory should look like:
```
models/fruit_model_pytorch/
├── model.pth          # Your trained PyTorch model
├── labels.txt         # Class names (one per line)
└── config.json        # Optional: model metadata
```

## Troubleshooting

### Common Issues:

1. **Model Loading Error**: Make sure you're saving the entire model, not just state dict
2. **Class Name Mismatch**: Ensure labels.txt matches your training classes
3. **Input Size Mismatch**: Update the transform to match your training input size
4. **CUDA Error**: The model service automatically handles CPU/GPU detection

### Debug Mode:
The model service prints loading status. Check the Flask console for messages like:
```
[ModelService] PyTorch model loaded successfully
[ModelService] Labels: ['fresh_apple', 'fresh_banana', ...]
```

## Testing Your Integration

1. **Upload Test**: Try uploading a fruit image via the web interface
2. **Camera Test**: Use the live camera feature to test real-time predictions
3. **Check Logs**: Monitor the Flask console for any error messages

Your PyTorch model is now fully integrated with the web application! 🎉
