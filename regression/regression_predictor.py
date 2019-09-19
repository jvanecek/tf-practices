
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

class RegressionPredictor:

    _model = ''

    def _buildSequentialModel(self, input_shape_size ):
        self._model = keras.Sequential([
            layers.Dense(64, activation=tf.nn.relu, input_shape=[input_shape_size]),
            layers.Dense(64, activation=tf.nn.relu),
            layers.Dense(1)
        ])

        optimizer = tf.keras.optimizers.RMSprop(0.001)

        self._model.compile(
            loss='mean_squared_error',
            optimizer=optimizer,
            metrics=['mean_absolute_error', 'mean_squared_error'])

    def train( self, normalized_training_data, training_labels, epochs, callbacks ):
        self._buildSequentialModel( len( normalized_training_data.keys() ) )

        return self._model.fit(
            normalized_training_data, 
            training_labels,
            epochs=epochs, 
            validation_split = 0.2, 
            verbose=0,
            callbacks=callbacks)

    def evaluateModelAccuracy(self, normalized_testing_data, testing_labels):
        return self._model.evaluate( normalized_testing_data, testing_labels, verbose=0 )

    def predict( self, normed_test_data ):
        return self._model.predict(normed_test_data).flatten()