"""
Fruit Ripeness Classification Training Script
Uses TensorFlow/Keras with transfer learning (MobileNetV2)
"""

import os
import pathlib
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_INITIAL = 10
EPOCHS_FINETUNE = 5
LEARNING_RATE_INITIAL = 0.001
LEARNING_RATE_FINETUNE = 0.0001

def create_dataset(data_dir, img_size=IMG_SIZE, batch_size=BATCH_SIZE, validation_split=None, subset=None):
    """Create dataset from directory structure"""
    return tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        image_size=img_size,
        batch_size=batch_size,
        validation_split=validation_split,
        subset=subset,
        seed=42
    )

def prepare_dataset(ds):
    """Optimize dataset performance"""
    AUTOTUNE = tf.data.AUTOTUNE
    return ds.prefetch(AUTOTUNE)

def create_data_augmentation():
    """Create data augmentation layer"""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ])

def create_model(num_classes, img_size=IMG_SIZE):
    """Create transfer learning model with MobileNetV2"""
    # Base model
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=img_size + (3,),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # Freeze base model initially
    
    # Data augmentation
    data_augmentation = create_data_augmentation()
    
    # Model architecture
    inputs = tf.keras.Input(shape=img_size + (3,))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = data_augmentation(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    return model, base_model

def plot_training_history(history, save_path=None):
    """Plot training metrics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def main():
    print("🍎 Fruit Ripeness Classification Training")
    print("=" * 50)
    
    # Check data directory
    data_dir = pathlib.Path("data/images")
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    
    if not train_dir.exists():
        print(f"❌ Training directory not found: {train_dir}")
        print("Please create the data structure and add images:")
        print("data/images/train/[class_name]/image.jpg")
        return
    
    # Option 1: Use separate train/val directories if they exist
    if val_dir.exists() and any(val_dir.iterdir()):
        print("📁 Using separate train/val directories")
        train_ds = create_dataset(train_dir)
        val_ds = create_dataset(val_dir)
    else:
        # Option 2: Split training data automatically
        print("📁 Splitting training data (80/20)")
        train_ds = create_dataset(train_dir, validation_split=0.2, subset="training")
        val_ds = create_dataset(train_dir, validation_split=0.2, subset="validation")
    
    # Get class names and info
    class_names = train_ds.class_names
    num_classes = len(class_names)
    
    print(f"📊 Found {num_classes} classes: {class_names}")
    
    # Prepare datasets
    train_ds = prepare_dataset(train_ds)
    val_ds = prepare_dataset(val_ds)
    
    # Create model
    print("🏗️ Creating model...")
    model, base_model = create_model(num_classes)
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_INITIAL),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("📋 Model Summary:")
    model.summary()
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-7
        )
    ]
    
    # Initial training (frozen base)
    print(f"🚀 Starting initial training ({EPOCHS_INITIAL} epochs)...")
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_INITIAL,
        callbacks=callbacks,
        verbose=1
    )
    
    # Fine-tuning (unfreeze some layers)
    print("🔧 Fine-tuning model...")
    base_model.trainable = True
    
    # Freeze early layers, fine-tune later layers
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_FINETUNE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Continue training
    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_FINETUNE,
        initial_epoch=len(history1.history['accuracy']),
        callbacks=callbacks,
        verbose=1
    )
    
    # Combine histories
    combined_history = {}
    for key in history1.history.keys():
        combined_history[key] = history1.history[key] + history2.history[key]
    
    # Create a mock history object for plotting
    class MockHistory:
        def __init__(self, history_dict):
            self.history = history_dict
    
    combined_history_obj = MockHistory(combined_history)
    
    # Plot results
    print("📈 Plotting training history...")
    plot_training_history(combined_history_obj, 'training_history.png')
    
    # Evaluate on validation set
    print("📊 Final evaluation:")
    val_loss, val_accuracy = model.evaluate(val_ds, verbose=0)
    print(f"Validation accuracy: {val_accuracy:.4f}")
    print(f"Validation loss: {val_loss:.4f}")
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = pathlib.Path(f"models/fruit_model_{timestamp}")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save in SavedModel format
    model.save(model_dir)
    print(f"💾 Model saved to: {model_dir}")
    
    # Save labels
    labels_path = model_dir / "labels.txt"
    with open(labels_path, 'w', encoding='utf-8') as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")
    print(f"🏷️ Labels saved to: {labels_path}")
    
    # Update .env with new model path
    env_path = pathlib.Path(".env")
    if env_path.exists():
        with open(env_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Update MODEL_PATH line
        with open(env_path, 'w', encoding='utf-8') as f:
            for line in lines:
                if line.startswith('MODEL_PATH='):
                    f.write(f"MODEL_PATH={model_dir}\n")
                else:
                    f.write(line)
        print(f"🔧 Updated .env with MODEL_PATH={model_dir}")
    
    print("✅ Training completed successfully!")
    print(f"🎯 To use this model, restart your Flask app:")
    print(f"   .\venv\Scripts\python app.py")

if __name__ == "__main__":
    main()
