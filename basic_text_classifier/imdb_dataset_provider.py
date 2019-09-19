import numpy as np
from tensorflow import keras

class IMDBDatasetProvider: 
    _PADDING_CHAR = 0

    def __init__(self, words_in_training_data):
        self._words_in_training_data = words_in_training_data
        self.imdb = keras.datasets.imdb
        (self._raw_train_data, self._train_labels), (self._raw_test_data, self._test_labels)  = self.imdb.load_data(num_words=words_in_training_data)
        # Format:
        # train_data: Array of Reviews. Each Review is a collection of numbers, which represents a word in a Dictionary
        # train_labels: Array of 0 (Negative) and 1 (Positive)

    def paddedTrainingSet(self): 
        return ( self._padded( self._raw_train_data ), self._train_labels )
        
    def paddedTestingSet(self): 
        return ( self._padded( self._raw_test_data ), self._test_labels )

    def multiHotTrainingSet(self): 
        return ( self._as_multi_hot( self._raw_train_data ), self._train_labels )

    def multiHotTestingSet(self): 
        return ( self._as_multi_hot( self._raw_test_data ), self._test_labels )

    def _padded(self, sequences):
        return keras.preprocessing.sequence.pad_sequences(sequences,
                                                            value=self._PADDING_CHAR,
                                                            padding='post',
                                                            maxlen=256)

    def _as_multi_hot(self, sequences):
        # Create an all-zero matrix of shape (len(sequences), dimension)
        results = np.zeros((len(sequences), self._words_in_training_data))
        for i, word_indices in enumerate(sequences):
            results[i, word_indices] = 1.0  # set specific indices of results[i] to 1s
        return results


    def _downloadWordIndex(self):
        # A dictionary mapping words to an integer index
        word_index = self.imdb.get_word_index()

        # The first indices are reserved
        word_index = {k:(v+3) for k,v in word_index.items()}
        word_index["<PAD>"] = self._PADDING_CHAR
        word_index["<START>"] = 1
        word_index["<UNK>"] = 2  # unknown
        word_index["<UNUSED>"] = 3

        return word_index

    def decode_review(self,text):
        reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
        return ' '.join([reverse_word_index.get(i, '?') for i in text])