import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
import seaborn as sns
cifar10_path = "/Users/admin/Downloads/cifar-10-batches-py"  # Change this to the path where you extracted the dataset
# Load the CIFAR-10 dataset
(X, Y), (X_test, Y_test) = cifar10.load_data()

# Class labels for CIFAR-10
class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Display a few images from the dataset
plt.figure(figsize=(15, 3))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(X[i])
    plt.title(class_labels[Y[i][0]])
    plt.axis('off')
plt.show()

# Normalize the data
scaler = MinMaxScaler()
X_scaled = X.reshape(-1, 32*32*3).astype(float)
X_scaled = scaler.fit_transform(X_scaled).reshape(-1, 32, 32, 3)
X_test_scaled = X_test.reshape(-1, 32*32*3).astype(float)
X_test_scaled = scaler.transform(X_test_scaled).reshape(-1, 32, 32, 3)

# Split the data into training, validation, and testing sets
X_temp, X_test_scaled, Y_temp, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X_temp, Y_temp, test_size=0.25, random_state=42)

# Define the CNN model
model = Sequential([
    Conv2D(32, (5, 5),padding='same', activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(32, (5, 5),padding='same', activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Show model summary
model.summary()

# Train the model
history = model.fit(X_train, Y_train, epochs=20, validation_data=(X_val, Y_val))

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and validation loss graph')
plt.legend()
plt.show()

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test_scaled, Y_test, verbose=0)

# Predict on test set
y_pred = model.predict(X_test_scaled)
y_pred_labels = np.argmax(y_pred, axis=1)
conf_matrix = confusion_matrix(Y_test, y_pred_labels)

# Display confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Reds', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Calculate precision and recall
precision = precision_score(Y_test, y_pred_labels, average='weighted')
recall = recall_score(Y_test, y_pred_labels, average='weighted')

# Output accuracy, precision, and recall
print(f"Train accuracy: {history.history['accuracy'][-1]}")
print(f"Test accuracy: {test_accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")


learning_rates = [0.0001,0.001,0.01,0.1]
train_losses = []
val_losses = []

for lr in learning_rates:
    print(f"Training model with learning rate: {lr}")
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, Y_train, epochs=20, validation_data=(X_val, Y_val), verbose=1)
    
    train_losses.append(history.history['loss'])
    val_losses.append(history.history['val_loss'])

plt.figure(figsize=(10, 6))
for i, lr in enumerate(learning_rates):
    plt.plot(train_losses[i], label=f'Training Loss (LR={lr})')
    plt.plot(val_losses[i], label=f'Validation Loss (LR={lr})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss for Different Learning Rates')
    plt.legend()
    plt.show()