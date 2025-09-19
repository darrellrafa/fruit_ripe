import os
from typing import Dict, Any
from PIL import Image
import numpy as np
from pathlib import Path
import json


class FruitRipenessModel:
    """
    Placeholder model service for fruit ripeness.
    Replace the simple heuristic with your deep learning model.
    """

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path
        self.model = None
        self.labels: list[str] | None = None
        self.model_type = None  # 'tensorflow', 'pytorch', or None
        self.device = None
        self.transform = None

        # Try to load model if path is provided
        if model_path:
            self._load_model(model_path)

    def _load_model(self, model_path: str):
        """Load either TensorFlow or PyTorch model"""
        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            print(f"[ModelService] MODEL_PATH does not exist: {model_path}")
            return

        # Check for PyTorch model files
        pytorch_files = list(model_path_obj.glob("*.pth")) + list(model_path_obj.glob("*.pt"))
        tensorflow_files = list(model_path_obj.glob("saved_model.pb")) or model_path_obj.is_dir()

        if pytorch_files:
            self._load_pytorch_model(model_path_obj, pytorch_files[0])
        elif tensorflow_files:
            self._load_tensorflow_model(model_path_obj)
        else:
            print(f"[ModelService] No recognized model files found in {model_path}")

    def _load_pytorch_model(self, model_dir: Path, model_file: Path):
        """Load PyTorch model"""
        try:
            import torch
            import torchvision.transforms as transforms
            
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"[ModelService] Using device: {self.device}")
            
            # Load model
            self.model = torch.load(model_file, map_location=self.device)
            self.model.eval()
            self.model_type = 'pytorch'
            
            # Load config if available
            config_path = model_dir / "config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    self.labels = config.get('labels', [])
                    # You can add more config parameters here
            
            # Load labels from labels.txt if config doesn't exist
            if not self.labels:
                labels_path = model_dir / "labels.txt"
                if labels_path.exists():
                    with open(labels_path, "r", encoding="utf-8") as f:
                        self.labels = [line.strip() for line in f if line.strip()]
            
            # Load transform settings from config or use defaults
            input_size = 224
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                        if 'input_size' in config:
                            input_size = config['input_size'][0] if isinstance(config['input_size'], list) else config['input_size']
                        if 'preprocessing' in config:
                            mean = config['preprocessing'].get('mean', mean)
                            std = config['preprocessing'].get('std', std)
                except Exception as e:
                    print(f"[ModelService] Warning: Could not load config preprocessing: {e}")
            
            self.transform = transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
            
            print(f"[ModelService] Using input size: {input_size}x{input_size}")
            
            print(f"[ModelService] PyTorch model loaded successfully")
            if self.labels:
                print(f"[ModelService] Labels: {self.labels}")
                
        except Exception as e:
            print(f"[ModelService] Failed to load PyTorch model: {e}")
            self.model = None

    def _load_tensorflow_model(self, model_path_obj: Path):
        """Load TensorFlow model"""
        try:
            import tensorflow as tf  # type: ignore
            self.model = tf.keras.models.load_model(model_path_obj)
            self.model_type = 'tensorflow'
            
            # Load labels
            labels_path = model_path_obj / "labels.txt"
            if labels_path.exists():
                with open(labels_path, "r", encoding="utf-8") as f:
                    self.labels = [line.strip() for line in f if line.strip()]
            
            print(f"[ModelService] TensorFlow model loaded successfully")
            if self.labels:
                print(f"[ModelService] Labels: {self.labels}")
                
        except Exception as e:
            print(f"[ModelService] Failed to load TensorFlow model: {e}")
            self.model = None

    def _preprocess(self, image_path: str) -> np.ndarray:
        """Preprocess image for model prediction. Adjust to match your model input."""
        img = Image.open(image_path).convert("RGB")
        # Resize to a standard size expected by your model (placeholder: 224x224)
        img = img.resize((224, 224))
        arr = np.asarray(img).astype(np.float32) / 255.0
        return arr

    def _simple_heuristic(self, image_path: str) -> Dict[str, Any]:
        """
        A simple color-based heuristic to simulate predictions.
        - If yellow-ish dominance: fruit_type=banana, ripe
        - If red dominance: apple, ripe
        - If green dominance: banana or apple not ripe
        """
        img = Image.open(image_path).convert("RGB").resize((128, 128))
        arr = np.asarray(img).astype(np.float32) / 255.0
        mean_rgb = arr.reshape(-1, 3).mean(axis=0)
        r, g, b = mean_rgb

        # Determine color dominance
        if r > 0.45 and g > 0.45 and b < 0.35:
            fruit_type = "banana"
            prediction = "ripe"
            confidence = float(min(0.99, 0.6 + 0.4 * (r + g) / 2))
        elif r > g and r > b and r > 0.4:
            fruit_type = "apple"
            prediction = "ripe"
            confidence = float(min(0.95, 0.55 + 0.4 * r))
        elif g > r and g > b:
            fruit_type = "banana"
            prediction = "not ripe"
            confidence = float(min(0.9, 0.5 + 0.4 * g))
        else:
            fruit_type = "fruit"
            prediction = "not sure"
            confidence = 0.5

        return {
            "fruit_type": fruit_type,
            "prediction": prediction,
            "confidence": confidence,
        }

    def predict(self, image_path: str) -> Dict[str, Any]:
        """Run prediction and return a dict with keys: fruit_type, prediction, confidence."""
        if self.model is None:
            # Use heuristic while actual model is not integrated
            return self._simple_heuristic(image_path)

        if self.model_type == 'pytorch':
            return self._predict_pytorch(image_path)
        elif self.model_type == 'tensorflow':
            return self._predict_tensorflow(image_path)
        else:
            return self._simple_heuristic(image_path)

    def _predict_pytorch(self, image_path: str) -> Dict[str, Any]:
        """PyTorch prediction"""
        try:
            import torch
            
            # Load and preprocess image
            img = Image.open(image_path).convert("RGB")
            if self.transform:
                img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            else:
                # Fallback transform
                img = img.resize((224, 224))
                img_array = np.array(img).astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(img_tensor)
                if hasattr(outputs, 'logits'):
                    outputs = outputs.logits
                probs = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probs, 1)
                
                confidence = float(confidence.cpu().numpy()[0])
                top_idx = int(predicted.cpu().numpy()[0])
            
            # Get label
            if self.labels and 0 <= top_idx < len(self.labels):
                label = self.labels[top_idx]
            else:
                label = str(top_idx)
            
            return self._parse_prediction_result(label, confidence, top_idx)
            
        except Exception as e:
            print(f"[ModelService] PyTorch prediction error: {e}")
            return self._simple_heuristic(image_path)

    def _predict_tensorflow(self, image_path: str) -> Dict[str, Any]:
        """TensorFlow prediction"""
        try:
            x = self._preprocess(image_path)
            x = np.expand_dims(x, axis=0)

            # Some models output logits, others probabilities
            preds = self.model.predict(x)[0]
            preds = np.asarray(preds).astype(np.float32)
            
            # Convert to probabilities if needed
            def softmax(z: np.ndarray) -> np.ndarray:
                z = z - np.max(z)
                exp_z = np.exp(z)
                return exp_z / np.sum(exp_z)

            if preds.ndim == 0:
                preds = np.array([preds])
            if np.any(preds < 0) or np.any(preds > 1) or not np.isclose(np.sum(preds), 1.0, atol=1e-3):
                probs = softmax(preds)
            else:
                probs = preds

            top_idx = int(np.argmax(probs))
            confidence = float(probs[top_idx])

            # Get label
            if self.labels and 0 <= top_idx < len(self.labels):
                label = self.labels[top_idx]
            else:
                label = str(top_idx)

            return self._parse_prediction_result(label, confidence, top_idx)
            
        except Exception as e:
            print(f"[ModelService] TensorFlow prediction error: {e}")
            return self._simple_heuristic(image_path)

    def _parse_prediction_result(self, label: str, confidence: float, top_idx: int) -> Dict[str, Any]:
        """Parse prediction result into fruit type and ripeness"""
        # Attempt to parse label into fruit type and ripeness
        fruit_type = "fruit"
        ripeness = "not sure"
        parts = label.lower().replace('-', '_').split('_')
        
        if len(parts) >= 2:
            fruit_type = parts[0]
            if "ripe" in parts:
                # Determine not ripe vs ripe based on tokens
                ripeness = "not ripe" if any(p in ("unripe", "green", "not") for p in parts) else "ripe"
            elif any(p in ("unripe", "green") for p in parts):
                ripeness = "not ripe"
        else:
            # If label doesn't encode ripeness, set based on threshold
            ripeness = "ripe" if confidence >= 0.6 else "not sure"

        return {
            "fruit_type": fruit_type,
            "prediction": ripeness,
            "confidence": confidence,
            "raw_label": label,
            "top_index": top_idx,
        }
