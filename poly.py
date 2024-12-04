import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense
x = np.random.randint(-20, 20, 10000)
y = 5 * x ** 3 - 8 * x ** 2 - 7 * x + 1
x_normalized = (2 * (x - np.min(x)) / (np.max(x) - np.min(x))) -1
y_normalized = (2 * (y - np.min(y)) / (np.max(y) - np.min(y))) -1
size = len(x_normalized)
x_train = x_normalized[:int(size * 0.9)]
y_train = y_normalized[:int(size * 0.9)]
x_test = x_normalized[int(size * 0.95):]
y_test = y_normalized[int(size * 0.95):]
x_validation = x_normalized[int(size * 0.9):int(size * 0.95)]
y_validation = y_normalized[int(size * 0.9):int(size * 0.95)]
plt.figure(figsize=(16, 6))
plt.scatter(x_train, y_train, label='Training data')
plt.scatter(x_test, y_test, label='Test data')
plt.scatter(x_validation, y_validation, label='Validation data')
plt. legend()
plt.grid()
plt.show()
inputs = Input((1,))
hidden = Dense(32, activation='sigmoid')(inputs)
hidden = Dense(64, activation='sigmoid')(hidden)
hidden = Dense(128, activation='sigmoid')(hidden)
outputs = Dense(1, name='Output_layer')(hidden)

model = Model(inputs, outputs, name = 'DNN')
model.summary()

model.compile(optimizer = "adam", loss = "mean_squared_error", metrics = [tf.keras.metrics.R2Score(name = "accuracy")])
model_fit = model.fit(x_train, y_train, epochs=50, validation_data=(x_validation, y_validation), verbose=0)
history = model_fit.history


plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
plt.plot(history['accuracy'],label='Training accuracy')
plt.plot(history['val_accuracy'],label='Validation accuracy')
plt.title('Training accuracy vs Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(history['loss'],label='Training loss')
plt.plot(history['val_loss'],label='Validation loss')
plt.title('Training loss vs Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()

plt.show()

predictions = model.predict(x_test)

plt.figure(figsize=(16, 6))
plt.scatter(x_test, y_test, label='True level')
plt.scatter(x_test, predictions, label='Prediction')
plt.legend()
plt.grid()
plt.show()
