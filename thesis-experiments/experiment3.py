import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import SparseCategoricalCrossentropy

# Define the model
model_dense = Sequential([
    Flatten(input_shape=(28, 28)),  # Assuming input shape is 28x28 (e.g., MNIST dataset)
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')  # Assuming 10 classes for classification
])

# Compile the model
model_dense.compile(optimizer=Adam(learning_rate=0.001),
                    loss=SparseCategoricalCrossentropy(),
                    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

# Summary of the model
model_dense.summary()

# Load the dataset (using MNIST as an example)
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the data
x_train = x_train / 255.0
x_test = x_test / 255.0

log_dir = "logs/experiment-3/python"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_steps_per_second=True)

history = model_dense.fit(
    x=x_train,
    y=y_train,
    epochs=10,
    batch_size=32,
    validation_data=(x_test, y_test),
    callbacks=[
      tensorboard_callback
    ]
)

# Evaluate the fully connected model
dense_loss, dense_accuracy = model_dense.evaluate(x_test, y_test)
print(f'Dense Model - Loss: {dense_loss}, Accuracy: {dense_accuracy}')
