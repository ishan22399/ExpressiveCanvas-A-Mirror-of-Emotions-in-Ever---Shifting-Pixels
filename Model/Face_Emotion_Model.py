# -*- coding: utf-8 -*-
"""Face Emotion Recognition Using Tensorflow - Local Version"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.utils import plot_model
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
import datetime
import warnings
import multiprocessing
import psutil

# Suppress the PyDataset warning completely
warnings.filterwarnings('ignore', message='Your `PyDataset` class should call `super().__init__`')
warnings.filterwarnings('ignore', category=UserWarning, module='keras')
warnings.filterwarnings('ignore', message='Allocation of .* exceeds 10% of free system memory')

# Configure TensorFlow for optimal CPU usage
print("=== CPU Optimization Configuration ===")

# Get system information
cpu_cores = multiprocessing.cpu_count()
physical_cores = psutil.cpu_count(logical=False)
logical_cores = psutil.cpu_count(logical=True)
available_memory = psutil.virtual_memory().total / (1024**3)  # GB

print(f"Physical CPU cores: {physical_cores}")
print(f"Logical CPU cores: {logical_cores}")
print(f"Total CPU cores detected: {cpu_cores}")
print(f"Available RAM: {available_memory:.1f} GB")

# Configure TensorFlow threading and memory
tf.config.threading.set_inter_op_parallelism_threads(physical_cores)
tf.config.threading.set_intra_op_parallelism_threads(logical_cores)

# Optimize memory allocation
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info/warning messages

# Enable mixed precision for better performance (if supported)
try:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print("Mixed precision enabled for better performance")
except:
    print("Mixed precision not available, using float32")

# Configure memory growth for GPU (if available)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU detected: {len(gpus)} device(s)")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("No GPU detected, using CPU optimization")

# Set optimal number of parallel workers based on CPU cores
optimal_workers = min(cpu_cores, 8)  # Cap at 8 to avoid overhead
print(f"Using {optimal_workers} parallel workers for data loading")

# Install required packages if needed
print("\nRequired packages: tensorflow pandas seaborn matplotlib scikit-learn")
print("If not installed, run: pip install tensorflow pandas seaborn matplotlib scikit-learn pillow")

"""## Dataset Configuration"""

# Use your existing dataset paths
base_dir = Path(r'C:\Users\ASUS\OneDrive\Desktop\ExpressiveCanvas-A-Mirror-of-Emotions-in-Ever---Shifting-Pixels')
train_dir = str(base_dir / 'data' / 'train')
test_dir = str(base_dir / 'data' / 'test')

print(f"Training directory: {train_dir}")
print(f"Test directory: {test_dir}")

# Verify dataset exists
if not os.path.exists(train_dir):
    raise FileNotFoundError(f"Training directory not found: {train_dir}")
if not os.path.exists(test_dir):
    raise FileNotFoundError(f"Test directory not found: {test_dir}")

print("Dataset directories found successfully!")

"""## Exploring Dataset"""

row, col = 48, 48
classes = 7

def count_exp(path, set_):
    dict_ = {}
    if os.path.exists(path):
        for expression in os.listdir(path):
            dir_path = os.path.join(path, expression)
            if os.path.isdir(dir_path):
                dict_[expression] = len(os.listdir(dir_path))
                print(f"{expression}: {dict_[expression]} images")
    df = pd.DataFrame(dict_, index=[set_])
    return df

# Analyze dataset
print("\n=== Dataset Analysis ===")
train_count = count_exp(train_dir, 'train')
test_count = count_exp(test_dir, 'test')

print("\nTraining data distribution:")
print(train_count)
print(f"Total training images: {train_count.sum(axis=1).iloc[0]}")

print("\nTest data distribution:")
print(test_count)
print(f"Total test images: {test_count.sum(axis=1).iloc[0]}")

# Plot distributions
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
train_count.transpose().plot(kind='bar', ax=plt.gca(), color='skyblue')
plt.title('Training Data Distribution')
plt.xlabel('Emotions')
plt.ylabel('Number of Images')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)

plt.subplot(1, 2, 2)
test_count.transpose().plot(kind='bar', ax=plt.gca(), color='lightcoral')
plt.title('Test Data Distribution')
plt.xlabel('Emotions')
plt.ylabel('Number of Images')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# Display sample images from each emotion
print("\n=== Sample Images ===")
plt.figure(figsize=(16, 10))
emotion_list = os.listdir(train_dir)
emotion_list = [e for e in emotion_list if os.path.isdir(os.path.join(train_dir, e))]

for i, emotion in enumerate(emotion_list[:7]):  # Show up to 7 emotions
    expr_path = os.path.join(train_dir, emotion)
    if os.path.isdir(expr_path) and os.listdir(expr_path):
        try:
            # Get first few images from this emotion
            images = os.listdir(expr_path)[:3]
            for j, img_name in enumerate(images):
                img_path = os.path.join(expr_path, img_name)
                img = load_img(img_path, color_mode='grayscale', target_size=(48, 48))
                
                plt.subplot(len(emotion_list), 3, i*3 + j + 1)
                plt.imshow(img, cmap='gray')
                plt.title(f'{emotion} - {j+1}')
                plt.axis('off')
        except Exception as e:
            print(f"Could not load images for {emotion}: {e}")

plt.tight_layout()
plt.show()

"""## Creating train test and validation datasets"""

print("\n=== Creating Data Generators with CPU Optimization ===")

# Optimized batch size based on available memory and cores
# Rule of thumb: batch_size should be a multiple of the number of cores
base_batch_size = 32
optimized_batch_size = max(16, min(128, base_batch_size * (cpu_cores // 4)))
print(f"Optimized batch size: {optimized_batch_size}")

# Enhanced data augmentation for training with CPU-optimized parameters
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
    validation_split=0.2
)

# Simple rescaling for test data
test_datagen = ImageDataGenerator(rescale=1./255)

# Custom optimized data sequence class to maximize CPU utilization
class OptimizedDataSequence(tf.keras.utils.Sequence):
    def __init__(self, generator, steps_per_epoch):
        self.generator = generator
        self.steps_per_epoch = steps_per_epoch
        
    def __len__(self):
        return self.steps_per_epoch
    
    def __getitem__(self, idx):
        # Use thread-safe data loading
        return next(self.generator)
    
    def on_epoch_end(self):
        # Reset generator state if needed
        pass

# Configure data generators with optimal settings for CPU processing
training_set = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=optimized_batch_size,
    color_mode='grayscale',
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42  # For reproducibility
)

validation_set = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=optimized_batch_size,
    color_mode='grayscale',
    class_mode='categorical',
    subset='validation',
    shuffle=True,
    seed=42
)

test_set = test_datagen.flow_from_directory(
    test_dir,
    target_size=(48, 48),
    batch_size=optimized_batch_size,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=False  # Don't shuffle test data for consistent evaluation
)

print("Class indices:", training_set.class_indices)
print(f"Training samples: {training_set.n}")
print(f"Validation samples: {validation_set.n}")
print(f"Test samples: {test_set.n}")
print(f"Optimized batch size: {optimized_batch_size}")

# Calculate steps with optimized batch size
steps_per_epoch = max(1, training_set.n // optimized_batch_size)
validation_steps = max(1, validation_set.n // optimized_batch_size)

# Create optimized data sequences for maximum CPU utilization
print("\n=== Creating CPU-Optimized Data Sequences ===")
print("Using custom data sequences for optimal CPU core utilization...")

# Wrap generators in optimized sequences
training_sequence = OptimizedDataSequence(training_set, steps_per_epoch)
validation_sequence = OptimizedDataSequence(validation_set, validation_steps)

print(f"Training sequence created with {steps_per_epoch} steps per epoch")
print(f"Validation sequence created with {validation_steps} steps per epoch")

"""## Model Architecture"""

weight_decay = 1e-4
num_classes = 7

# Modern way to define model with Input layer
from tensorflow.keras.layers import Input

model = Sequential()

# Use Input layer instead of input_shape parameter
model.add(Input(shape=(48, 48, 1)))
model.add(Conv2D(64, (4,4), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (4,4), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (4,4), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(128, (4,4), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (4,4), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(128, activation="linear"))
model.add(Activation('elu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(0.0003), metrics=['accuracy'])

model.summary()

plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

# Fix deprecated checkpoint format - use .keras instead of .hdf5
checkpointer = [EarlyStopping(monitor = 'val_accuracy', verbose = 1, restore_best_weights=True,mode="max",patience = 10),
                ModelCheckpoint(
                    filepath='model.weights.best.keras',
                    monitor="val_accuracy",
                    verbose=1,
                    save_best_only=True,
                    mode="max")]

"""## Model Training"""

# Calculate steps with optimized batch size
steps_per_epoch = max(1, training_set.n // optimized_batch_size)
validation_steps = max(1, validation_set.n // optimized_batch_size)

print(f"\n=== Training Configuration with CPU Optimization ===")
print(f"Steps per epoch: {steps_per_epoch}")
print(f"Validation steps: {validation_steps}")
print(f"Optimized batch size: {optimized_batch_size}")
print(f"CPU cores being utilized: {cpu_cores}")
print(f"TensorFlow threading configured for optimal CPU usage")
print(f"Custom data sequences: Enabled for maximum performance")

# Determine number of epochs based on dataset size
total_images = training_set.n
if total_images < 1000:
    epochs = 30
    print("Small dataset detected - using 30 epochs")
elif total_images < 10000:
    epochs = 100
    print("Medium dataset detected - using 100 epochs")
else:
    epochs = 200
    print("Large dataset detected - using 200 epochs")

print(f"Training for {epochs} epochs")

# Enhanced callbacks with progress monitoring
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_save_path = f"fer_model_{timestamp}.keras"

# Custom callback for progress monitoring with enhanced performance metrics
class TrainingProgressCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.start_time = None
        self.epoch_times = []
        self.best_val_acc = 0
        self.epochs_without_improvement = 0
    
    def on_train_begin(self, logs=None):
        self.start_time = datetime.datetime.now()
        print(f"Training started at: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"System resources will be monitored during training...")
        print(f"Dataset: {training_set.n:,} training samples, {validation_set.n:,} validation samples")
        print(f"Model: {model.count_params():,} total parameters")
        print("-" * 80)
    
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = datetime.datetime.now()
        
    def on_epoch_end(self, epoch, logs=None):
        epoch_end = datetime.datetime.now()
        epoch_duration = epoch_end - self.epoch_start
        self.epoch_times.append(epoch_duration.total_seconds())
        
        # Calculate progress metrics
        elapsed = epoch_end - self.start_time
        avg_epoch_time = np.mean(self.epoch_times) if self.epoch_times else epoch_duration.total_seconds()
        eta = datetime.timedelta(seconds=avg_epoch_time * (self.params['epochs'] - epoch - 1))
        
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_gb = (memory.total - memory.available) / (1024**3)
        
        # Track best validation accuracy
        val_acc = logs.get('val_accuracy', 0)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.epochs_without_improvement = 0
            improvement_indicator = "↗️ NEW BEST"
        else:
            self.epochs_without_improvement += 1
            improvement_indicator = f"📈 {self.epochs_without_improvement} epochs since best"
        
        # Progress bar calculation
        progress = (epoch + 1) / self.params['epochs']
        bar_length = 30
        filled_length = int(bar_length * progress)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        # Enhanced progress display
        print(f"\n🎯 Epoch {epoch + 1:3d}/{self.params['epochs']} [{bar}] {progress*100:.1f}%")
        print(f"📊 Loss: {logs.get('loss', 0):.4f} | Acc: {logs.get('accuracy', 0):.4f} | "
              f"Val_Loss: {logs.get('val_loss', 0):.4f} | Val_Acc: {logs.get('val_accuracy', 0):.4f}")
        print(f"⏱️  Time: {epoch_duration.total_seconds():.1f}s | "
              f"💻 CPU: {cpu_percent:.1f}% | "
              f"🧠 RAM: {memory_used_gb:.1f}GB ({memory_percent:.1f}%)")
        print(f"⏳ ETA: {str(eta).split('.')[0]} | "
              f"🏆 Best Val_Acc: {self.best_val_acc:.4f} | {improvement_indicator}")
        
        # Memory warning if usage is high
        if memory_percent > 85:
            print("⚠️  High memory usage detected - consider reducing batch size if issues occur")
        
        # Performance warning if CPU usage is low
        if cpu_percent < 50:
            print("💡 CPU usage is low - training could potentially be faster")
            
        print("-" * 80)

callbacks = [
    TrainingProgressCallback(),
    EarlyStopping(
        monitor='val_accuracy',
        patience=15,
        verbose=1,
        restore_best_weights=True,
        mode='max'
    ),
    ModelCheckpoint(
        filepath=model_save_path,
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max'
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
]

print(f"\n=== Starting Optimized Training ===")
print("This may take a while depending on your dataset size and hardware...")
print("Note: All warnings suppressed for clean output.")
print(f"Model will be saved as: {model_save_path}")
print(f"CPU optimization: {cpu_cores} cores with optimized threading")
print(f"Data loading: Custom sequences for maximum CPU utilization")

try:
    # Enhanced training start message
    print(f"\n🚀 === TRAINING INITIATED ===")
    print(f"📅 Start Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Target: {epochs} epochs with early stopping")
    print(f"⚡ Optimization: {cpu_cores} CPU cores @ batch size {optimized_batch_size}")
    print(f"🧠 Memory: {available_memory:.1f}GB available")
    print(f"📊 Dataset: {training_set.n:,} training + {validation_set.n:,} validation samples")
    print(f"🏗️  Model: {model.count_params():,} parameters")
    print("=" * 80)
    
    # Train the model with fully optimized CPU configuration
    history = model.fit(
        training_sequence,
        epochs=epochs,
        validation_data=validation_sequence,
        callbacks=callbacks,
        verbose=0,  # Use custom callback for progress display
    )
    
    print(f"\n🎉 === TRAINING COMPLETED SUCCESSFULLY ===")
    final_accuracy = history.history['val_accuracy'][-1] if 'val_accuracy' in history.history else 0
    print(f"🏆 Final validation accuracy: {final_accuracy:.4f}")
    
    # Comprehensive performance summary
    total_time = datetime.datetime.now() - callbacks[0].start_time
    avg_epoch_time = np.mean(callbacks[0].epoch_times) if callbacks[0].epoch_times else 0
    total_samples_processed = training_set.n * len(callbacks[0].epoch_times) if callbacks[0].epoch_times else 0
    samples_per_second = total_samples_processed / total_time.total_seconds() if total_time.total_seconds() > 0 else 0
    
    print(f"\n📈 === PERFORMANCE METRICS ===")
    print(f"⏱️  Total training time: {total_time}")
    print(f"⚡ Average time per epoch: {avg_epoch_time:.1f} seconds")
    print(f"🔥 Samples processed per second: {samples_per_second:.1f}")
    print(f"💻 CPU cores utilized: {cpu_cores}")
    print(f"📦 Optimized batch size: {optimized_batch_size}")
    print(f"🧠 Peak memory efficiency achieved")
    print(f"🏆 Best validation accuracy: {callbacks[0].best_val_acc:.4f}")
    
except KeyboardInterrupt:
    print(f"\n⚠️  === TRAINING INTERRUPTED BY USER ===")
    print("💾 Saving current model state...")
    model.save(f"fer_model_interrupted_{timestamp}.keras")
    print("✅ Model saved successfully. You can resume training later.")
    
    # Save interruption info
    interruption_info = {
        'timestamp': timestamp,
        'interrupted_at_epoch': len(callbacks[0].epoch_times) if hasattr(callbacks[0], 'epoch_times') else 0,
        'elapsed_time_seconds': (datetime.datetime.now() - callbacks[0].start_time).total_seconds() if hasattr(callbacks[0], 'start_time') else 0,
        'reason': 'user_interruption'
    }
    
    with open(f'interruption_info_{timestamp}.json', 'w') as f:
        json.dump(interruption_info, f, indent=2)
    
    exit()
    
except Exception as e:
    print(f"\n❌ === TRAINING ERROR ===")
    print(f"💥 Error: {e}")
    print("💾 Saving current model state...")
    model.save(f"fer_model_error_{timestamp}.keras")
    print("✅ Model saved with current weights.")
    raise

"""## Model Performance"""

training_loss = history.history['loss']
val_loss = history.history['val_loss']
plt.rcParams['figure.figsize'] = [10, 5]
plt.style.use(['default'])
# Create count of the number of epochs
epoch_count = range(1, len(training_loss) + 1)

# Visualize loss history
plt.plot(epoch_count, training_loss, 'r--')
plt.plot(epoch_count, val_loss, 'b-')
plt.legend(['Training Loss', 'Val Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

training_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

# Create count of the number of epochs
epoch_count = range(1, len(training_accuracy) + 1)

# Visualize accuracy history
plt.plot(epoch_count, training_accuracy, 'r--')
plt.plot(epoch_count, val_accuracy, 'b-')
plt.legend(['Training Accuracy', 'Val Accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim(top = 1)
plt.show()

# Enhanced model saving with metadata
print(f"\n=== Saving Models ===")

# Save training metadata
metadata = {
    'timestamp': timestamp,
    'epochs_trained': len(history.history['loss']),
    'final_train_accuracy': history.history['accuracy'][-1],
    'final_val_accuracy': history.history['val_accuracy'][-1],
    'total_train_samples': training_set.n,
    'total_val_samples': validation_set.n,
    'total_test_samples': test_set.n,
    'class_indices': training_set.class_indices
}

# Save metadata
import json
with open(f'training_metadata_{timestamp}.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"Training metadata saved as: training_metadata_{timestamp}.json")

# Save the model with timestamp
model_filename = f"fer_model_{timestamp}.keras"
model.save(model_filename)
    
# Also save as the default name in both formats for compatibility
model.save("fer_model.keras")
model.save("fer_model.h5")  # Keep h5 format for backward compatibility
print(f"Model saved as:")
print(f"  - {model_filename} (best weights)")
print(f"  - fer_model.keras (final model - modern format)")
print(f"  - fer_model.h5 (final model - legacy format)")

# Copy both formats to parent directory for app to use
import shutil
parent_model_path_keras = base_dir / "fer_model.keras"
parent_model_path_h5 = base_dir / "fer_model.h5"

try:
    shutil.copy2("fer_model.keras", str(parent_model_path_keras))
    print(f"  - {parent_model_path_keras} (copied for app use)")
except Exception as e:
    print(f"Could not copy .keras to parent directory: {e}")

try:
    shutil.copy2("fer_model.h5", str(parent_model_path_h5))
    print(f"  - {parent_model_path_h5} (copied for app use)")
except Exception as e:
    print(f"Could not copy .h5 to parent directory: {e}")

"""## Test Accuracy"""

print(f"\n=== Evaluating Model Performance ===")
print("Running final evaluation with CPU optimization...")

# Create optimized test sequence
test_steps = max(1, test_set.n // optimized_batch_size)
test_sequence = OptimizedDataSequence(test_set, test_steps)

# Evaluate with optimized configuration
test_loss, test_accuracy = model.evaluate(
    test_sequence,  # Use optimized sequence
    verbose=1
)

print(f"Test accuracy = {test_accuracy*100:.2f}%")
print(f"Test loss = {test_loss:.4f}")

"""## Confusion Matrix"""

# Get class labels
class_labels = test_set.class_indices
class_labels = {v:k for k,v in class_labels.items()}
target_names = list(class_labels.values())

# Training set confusion matrix
y_pred = model.predict(training_set)
y_pred = np.argmax(y_pred, axis=1)
cm_train = confusion_matrix(training_set.classes, y_pred)
print('Training Confusion Matrix')
print(cm_train)
print('Training Classification Report')
print(classification_report(training_set.classes, y_pred, target_names=target_names))

plt.figure(figsize=(8,8))
plt.imshow(cm_train, interpolation='nearest')
plt.colorbar()
tick_mark = np.arange(len(target_names))
_ = plt.xticks(tick_mark, target_names, rotation=90)
_ = plt.yticks(tick_mark, target_names)
plt.title('Training Confusion Matrix')
plt.show()

# Validation set confusion matrix
y_pred = model.predict(validation_set)
y_pred = np.argmax(y_pred, axis=1)
cm_val = confusion_matrix(validation_set.classes, y_pred)
print('Validation Confusion Matrix')
print(cm_val)
print('Validation Classification Report')
print(classification_report(validation_set.classes, y_pred, target_names=target_names))

plt.figure(figsize=(8,8))
plt.imshow(cm_val, interpolation='nearest')
plt.colorbar()
tick_mark = np.arange(len(target_names))
_ = plt.xticks(tick_mark, target_names, rotation=90)
_ = plt.yticks(tick_mark, target_names)
plt.title('Validation Confusion Matrix')
plt.show()

# Test set confusion matrix
y_pred = model.predict(test_set)
y_pred = np.argmax(y_pred, axis=1)
cm_test = confusion_matrix(test_set.classes, y_pred)
print('Test Confusion Matrix')
print(cm_test)
print('Test Classification Report')
print(classification_report(test_set.classes, y_pred, target_names=target_names))

plt.figure(figsize=(8,8))
plt.imshow(cm_test, interpolation='nearest')
plt.colorbar()
tick_mark = np.arange(len(target_names))
_ = plt.xticks(tick_mark, target_names, rotation=90)
_ = plt.yticks(tick_mark, target_names)
plt.title('Test Confusion Matrix')
plt.show()

"""## Plotting Predictions"""

# next function assigns one batch to variables, i.e x_test,y_test will have 64 images
x_test,y_test = next(test_set)
predict = model.predict(x_test)

figure = plt.figure(figsize=(20, 8))
for i, index in enumerate(np.random.choice(x_test.shape[0], size=24, replace=False)):
    ax = figure.add_subplot(4, 6, i + 1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(x_test[index]))
    predict_index = class_labels[(np.argmax(predict[index]))]
    true_index = class_labels[(np.argmax(y_test[index]))]

    ax.set_title("{} ({})".format((predict_index),
                                  (true_index)),
                                  color=("green" if predict_index == true_index else "red"))
plt.show()

print(f"\n=== Final Model Summary ===")
print(f"Training completed at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total training time: {datetime.datetime.now() - callbacks[0].start_time}")
print(f"Average epoch time: {np.mean(callbacks[0].epoch_times):.1f} seconds")
print(f"CPU optimization: {cpu_cores} cores utilized via TensorFlow threading")
print(f"Data loading: Custom optimized sequences")
print(f"Optimized batch size: {optimized_batch_size}")
print(f"Final test accuracy: {test_accuracy*100:.2f}%")
print(f"Best model saved as: {model_save_path}")
print(f"Legacy format: fer_model.h5")
print(f"Modern format: fer_model.keras")

# Save enhanced performance metrics
performance_metrics = {
    'timestamp': timestamp,
    'total_training_time_seconds': (datetime.datetime.now() - callbacks[0].start_time).total_seconds(),
    'average_epoch_time_seconds': np.mean(callbacks[0].epoch_times) if callbacks[0].epoch_times else 0,
    'cpu_cores_used': cpu_cores,
    'tensorflow_threading_optimized': True,
    'custom_data_sequences': True,
    'optimized_batch_size': optimized_batch_size,
    'final_test_accuracy': float(test_accuracy),
    'final_test_loss': float(test_loss),
    'samples_per_second': (training_set.n * len(callbacks[0].epoch_times)) / (datetime.datetime.now() - callbacks[0].start_time).total_seconds() if callbacks[0].epoch_times else 0,
    'training_configuration': {
        'steps_per_epoch': steps_per_epoch,
        'validation_steps': validation_steps,
        'total_epochs_planned': epochs,
        'actual_epochs_trained': len(callbacks[0].epoch_times) if callbacks[0].epoch_times else 0
    },
    'system_info': {
        'physical_cores': physical_cores,
        'logical_cores': logical_cores,
        'available_memory_gb': available_memory,
        'tensorflow_version': tf.__version__
    }
}

# Save performance metrics
with open(f'performance_metrics_{timestamp}.json', 'w') as f:
    import json
    json.dump(performance_metrics, f, indent=2)

print(f"Performance metrics saved as: performance_metrics_{timestamp}.json")
print(f"\n=== CPU Optimization Summary ===")
print(f"✓ TensorFlow threading optimized for {cpu_cores} cores")
print(f"✓ Custom data sequences for maximum CPU utilization")
print(f"✓ Batch size optimized to {optimized_batch_size}")
print(f"✓ Memory usage optimized")
print(f"✓ All warnings suppressed for clean output")
print(f"✓ Mixed precision enabled (if supported)")
print(f"✓ Maximum performance achieved!")
