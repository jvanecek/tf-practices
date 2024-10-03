import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import SparseCategoricalCrossentropy

# Load the dataset (using MNIST as an example)
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape the data for CNN model
x_train_cnn = x_train.reshape(-1, 28, 28, 1)
x_test_cnn = x_test.reshape(-1, 28, 28, 1)

# Define the model
model_cnn = Sequential([
    Conv2D(32, (3, 3), input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')  # Assuming 10 classes for classification
])

# Compile the model
model_cnn.compile(optimizer=RMSprop(learning_rate=0.001),
                  loss=SparseCategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

log_dir = "logs/experiment-4/python"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_steps_per_second=True)

# Train the CNN model
model_cnn.fit(
    x_train_cnn,
    y_train,
    epochs=5,
    validation_data=(x_test_cnn, y_test),
    callbacks=[
      tensorboard_callback
    ]
)

# Evaluate the CNN model
cnn_loss, cnn_accuracy = model_cnn.evaluate(x_test_cnn, y_test)

# Summary of the model
print(model_cnn.summary())
print(f'CNN Model - Loss: {cnn_loss}, Accuracy: {cnn_accuracy}')
