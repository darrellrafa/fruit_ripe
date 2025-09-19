"""
Add this code to the end of your existing PyTorch notebook
to export your trained model for the web application.

Copy and paste these cells into your notebook after training is complete.
"""

# ===== CELL 1: Export Model =====
import torch
import json
from pathlib import Path
import zipfile

# IMPORTANT: Replace 'model' with your actual trained model variable name
# For example, if your model variable is called 'net' or 'classifier', use that instead
your_trained_model = model  # <-- CHANGE THIS to your model variable name

# Create export directory
export_dir = Path("fruit_model_export")
export_dir.mkdir(exist_ok=True)

# Save the model
model_path = export_dir / "model.pth"
torch.save(your_trained_model, model_path)
print(f"✅ Model saved to: {model_path}")

# ===== CELL 2: Create Labels =====
# IMPORTANT: Update this list with your actual class names from your notebook
# Look for where you defined your classes or check your dataset structure
class_names = [
    "fresh_apple",
    "fresh_banana", 
    "fresh_orange",
    "rotten_apple",
    "rotten_banana",
    "rotten_orange"
]  # <-- UPDATE THIS with your actual class names

# Save labels
labels_path = export_dir / "labels.txt"
with open(labels_path, 'w') as f:
    for class_name in class_names:
        f.write(f"{class_name}\n")
print(f"✅ Labels saved to: {labels_path}")
print(f"Classes: {class_names}")

# ===== CELL 3: Create Config =====
# Update these values based on your model training
config = {
    "labels": class_names,
    "input_size": [224, 224],  # Update if you used different input size
    "num_classes": len(class_names),
    "model_type": "pytorch",
    "preprocessing": {
        "mean": [0.485, 0.456, 0.406],  # Standard ImageNet values
        "std": [0.229, 0.224, 0.225]   # Update if you used different normalization
    }
}

config_path = export_dir / "config.json"
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)
print(f"✅ Config saved to: {config_path}")

# ===== CELL 4: Test Loading =====
# Test that the saved model can be loaded
try:
    test_model = torch.load(model_path, map_location='cpu')
    test_model.eval()
    print("✅ Model loads successfully!")
    print(f"Model type: {type(test_model)}")
    
    # Test with dummy input
    dummy_input = torch.randn(1, 3, 224, 224)  # Batch size 1, 3 channels, 224x224
    with torch.no_grad():
        output = test_model(dummy_input)
        print(f"✅ Model inference works! Output shape: {output.shape}")
        print(f"Number of classes: {output.shape[1]}")
        
except Exception as e:
    print(f"❌ Error testing model: {e}")

# ===== CELL 5: Create Download Package =====
zip_path = "fruit_model_for_webapp.zip"
with zipfile.ZipFile(zip_path, 'w') as zipf:
    for file_path in export_dir.glob("*"):
        zipf.write(file_path, file_path.name)

print(f"📦 Model package created: {zip_path}")
print("\n🎯 Next steps:")
print("1. Download the zip file from your notebook environment")
print("2. Extract it to: c:\\Users\\PC\\CascadeProjects\\windsurf-project\\models\\fruit_model_pytorch\\")
print("3. Update .env file: MODEL_PATH=models/fruit_model_pytorch")
print("4. Restart Flask app: .\\venv\\Scripts\\python app.py")

# ===== CELL 6: Show File Contents =====
print("\n📁 Created files:")
for file_path in export_dir.glob("*"):
    print(f"  - {file_path.name}")
    if file_path.suffix == '.txt':
        print(f"    Content preview: {file_path.read_text()[:100]}...")

print(f"\n📊 Model summary:")
print(f"  - Classes: {len(class_names)}")
print(f"  - Model file size: {model_path.stat().st_size / (1024*1024):.1f} MB")
print(f"  - Ready for web app integration!")
