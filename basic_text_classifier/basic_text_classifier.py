import tensorflow as tf
from tensorflow import keras

class BasicTextClassifier:
    _model = ''

    def _buildSequentialModel(self, vocabularySize):

        self._model = keras.Sequential()
        self._model.add(keras.layers.Embedding(vocabularySize, 16))
        self._model.add(keras.layers.GlobalAveragePooling1D())
        self._model.add(keras.layers.Dense(16, activation=tf.nn.relu))
        self._model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

        self._model.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['acc'])

    def train(self, normalized_training_data, training_label, validate_on_first, epochs, batch_size):
        self._buildSequentialModel(vocabularySize=10000)

        # Create a Validation Set
        x_val = normalized_training_data[:validate_on_first]
        partial_x_train = normalized_training_data[validate_on_first:]

        y_val = training_label[:validate_on_first]
        partial_y_train = training_label[validate_on_first:]

        # Train the Model
        return self._model.fit(partial_x_train,
                            partial_y_train,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=(x_val, y_val),
                            verbose=1)

    def evaluateModelAccuracy(self, normalized_testing_data, testing_labels):
        return self._model.evaluate( normalized_testing_data, testing_labels )