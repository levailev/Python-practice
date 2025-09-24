import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

# --- load CIFAR-10 dataset ---
from tensorflow.keras.datasets import cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Normalize pixel values
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# --- CNN model ---
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')  # 10 classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# --- Train the model ---
history = model.fit(X_train, y_train, validation_split=0.2, epochs=10, batch_size=64)

# --- Evaluate ---
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

# --- Plot training history ---
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# --- Predict on new test images ---
class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

sample_idx = np.random.randint(0, X_test.shape[0])
sample_img = X_test[sample_idx]
sample_label = np.argmax(y_test[sample_idx])

pred = model.predict(np.expand_dims(sample_img, axis=0))
pred_class = np.argmax(pred[0])

plt.imshow(sample_img)
plt.title(f"True: {class_names[sample_label]}, Pred: {class_names[pred_class]}")
plt.show()