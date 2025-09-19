"""
Add this cell to the END of your existing notebook:
c:\Users\PC\CascadeProjects\windsurf-project\models\f1-0-98-fruits-ripeness-classification-torch.ipynb

This will export your trained rexnet_150 model for the web application.
"""

# ===== ADD THIS CELL TO YOUR NOTEBOOK =====

import torch
import timm
import json
import zipfile
from pathlib import Path

print("🚀 Exporting your rexnet_150 model for web application...")

# Your existing variables (these should already be defined in your notebook)
# model_name = "rexnet_150"
# save_prefix = "fruits" 
# save_dir = "saved_models"
# classes = {...}  # Your classes dictionary
# device = "cuda" if torch.cuda.is_available() else "cpu"

# Create export directory
export_dir = Path("fruit_model_for_webapp")
export_dir.mkdir(exist_ok=True)

# Recreate and load your trained model
print("📦 Loading your trained model...")
model = timm.create_model(model_name=model_name, pretrained=False, num_classes=len(classes)).to(device)
model.load_state_dict(torch.load(f"{save_dir}/{save_prefix}_best_model.pth", map_location=device))
model.eval()

# Save the complete model (not just state dict)
model_path = export_dir / "model.pth"
torch.save(model, model_path)
print(f"✅ Model saved to: {model_path}")

# Create labels.txt from your classes
class_names = list(classes.keys())
labels_path = export_dir / "labels.txt"
with open(labels_path, 'w') as f:
    for class_name in class_names:
        f.write(f"{class_name}\n")
print(f"✅ Labels saved to: {labels_path}")
print(f"📊 Classes found: {class_names}")

# Create config.json with your model details
config = {
    "model_name": model_name,
    "labels": class_names,
    "input_size": [64, 64],  # Based on your im_size = 64
    "num_classes": len(classes),
    "model_type": "pytorch",
    "architecture": "rexnet_150",
    "preprocessing": {
        "mean": [0.485, 0.456, 0.406],  # Your mean values
        "std": [0.229, 0.224, 0.225]    # Your std values
    },
    "save_prefix": save_prefix,
    "original_save_dir": save_dir
}

config_path = export_dir / "config.json"
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)
print(f"✅ Config saved to: {config_path}")

# Test the exported model
print("🧪 Testing exported model...")
try:
    test_model = torch.load(model_path, map_location='cpu')
    test_model.eval()
    
    # Test with dummy input (64x64 based on your notebook)
    dummy_input = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        output = test_model(dummy_input)
        print(f"✅ Model works! Output shape: {output.shape}")
        print(f"✅ Number of classes: {output.shape[1]} (expected: {len(classes)})")
        
        # Test softmax probabilities
        probs = torch.softmax(output, dim=1)
        pred_class = torch.argmax(probs, dim=1)
        confidence = torch.max(probs, dim=1)[0]
        print(f"✅ Sample prediction: Class {pred_class.item()}, Confidence: {confidence.item():.3f}")
        
except Exception as e:
    print(f"❌ Error testing model: {e}")

# Create download package
zip_path = "rexnet_fruit_model.zip"
with zipfile.ZipFile(zip_path, 'w') as zipf:
    for file_path in export_dir.glob("*"):
        zipf.write(file_path, file_path.name)

print(f"\n📦 Model package created: {zip_path}")
print(f"📁 Files included:")
for file_path in export_dir.glob("*"):
    size_mb = file_path.stat().st_size / (1024*1024)
    print(f"  - {file_path.name} ({size_mb:.1f} MB)")

print(f"\n🎯 Next steps:")
print(f"1. Download {zip_path} from your notebook environment")
print(f"2. Extract to: models/fruit_rexnet_model/")
print(f"3. Update .env: MODEL_PATH=models/fruit_rexnet_model")
print(f"4. Restart Flask app")
print(f"\n✨ Your rexnet_150 model is ready for the web app!")

# Show class mapping for verification
print(f"\n📋 Class mapping:")
for i, class_name in enumerate(class_names):
    print(f"  {i}: {class_name}")
