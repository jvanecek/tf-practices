
import tensorflow as tf
from tensorflow import keras

class ClassifierBehavior: 
    def train( self, normalized_training_data, training_labels, validation_data, input_size ):
        self._buildSequentialModel(input_size)
        return self._model.fit(
            normalized_training_data, 
            training_labels,
            epochs=20,
            batch_size=512,
            validation_data=validation_data,
            verbose=2)

class BaselineClassifier(ClassifierBehavior):
    def _buildSequentialModel(self, input_size):
        self._model = keras.Sequential([
            # `input_shape` is only required here so that `.summary` works.
            keras.layers.Dense(16, activation=tf.nn.relu, input_shape=(input_size,)),
            keras.layers.Dense(16, activation=tf.nn.relu),
            keras.layers.Dense(1, activation=tf.nn.sigmoid)
        ])

        self._model.compile(optimizer='adam',
                            loss='binary_crossentropy',
                            metrics=['accuracy', 'binary_crossentropy'])

class SmallerModelClassifier(ClassifierBehavior):
    def _buildSequentialModel(self, input_size):
        self._model = keras.Sequential([
            keras.layers.Dense(4, activation=tf.nn.relu, input_shape=(input_size,)),
            keras.layers.Dense(4, activation=tf.nn.relu),
            keras.layers.Dense(1, activation=tf.nn.sigmoid)
        ])

        self._model.compile(optimizer='adam',
                        loss='binary_crossentropy',
                        metrics=['accuracy', 'binary_crossentropy'])

class BiggerModelClassifier(ClassifierBehavior): 
    def _buildSequentialModel(self, input_size):
        self._model = keras.models.Sequential([
            keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(input_size,)),
            keras.layers.Dense(512, activation=tf.nn.relu),
            keras.layers.Dense(1, activation=tf.nn.sigmoid)
        ])

        self._model.compile(optimizer='adam',
                            loss='binary_crossentropy',
                            metrics=['accuracy','binary_crossentropy'])


class L2RegularizedClassifier(ClassifierBehavior):
    def _buildSequentialModel(self, input_size):
        self._model = keras.models.Sequential([
            keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001),
                            activation=tf.nn.relu, input_shape=(input_size,)),
            keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001),
                            activation=tf.nn.relu),
            keras.layers.Dense(1, activation=tf.nn.sigmoid)
        ])

        self._model.compile(optimizer='adam',
                        loss='binary_crossentropy',
                        metrics=['accuracy', 'binary_crossentropy'])

class DropoutRegularizedClassifier(ClassifierBehavior): 
    def _buildSequentialModel(self, input_size): 
        self._model = keras.models.Sequential([
            keras.layers.Dense(16, activation=tf.nn.relu, input_shape=(input_size,)),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(16, activation=tf.nn.relu),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(1, activation=tf.nn.sigmoid)
        ])

        self._model.compile(optimizer='adam',
                        loss='binary_crossentropy',
                        metrics=['accuracy','binary_crossentropy'])