import pandas as pd
from tensorflow import keras

class AutoMPGDatasetProvider: 
    TRAINING_SET = ([],[])
    TESTING_SET = ([],[])
    
    def __init__(self): 
                
        dataset_path = keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")

        column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                        'Acceleration', 'Model Year', 'Origin']
        raw_dataset = pd.read_csv(dataset_path, names=column_names,
                            na_values = "?", comment='\t',
                            sep=" ", skipinitialspace=True)

        # Clean the data
        # The dataset contains a few unknown values. To keep this initial tutorial simple drop those rows.
        dataset = raw_dataset.copy().dropna()

        # The "Origin" column is really categorical, not numeric. So convert that to a one-hot:
        origin = dataset.pop('Origin')
        dataset['USA'] = (origin == 1)*1.0
        dataset['Europe'] = (origin == 2)*1.0
        dataset['Japan'] = (origin == 3)*1.0

        # Now split the dataset into a training set and a test set. 
        # We will use the test set in the final evaluation of our model.
        train_dataset = dataset.sample(frac=0.8,random_state=0)
        test_dataset = dataset.drop(train_dataset.index)

        train_stats = train_dataset.describe()
        train_stats.pop("MPG")
        train_stats = train_stats.transpose()

        # Separate the target value, or "label", from the features. 
        # This label is the value that you will train the model to predict.
        train_labels = train_dataset.pop('MPG')
        test_labels = test_dataset.pop('MPG')

        # It is good practice to normalize features that use different scales and ranges. 
        # Although the model might converge without feature normalization, 
        # it makes training more difficult, and it makes the resulting model dependent 
        # on the choice of units used in the input.
        normed_train_data = self._normalize(train_dataset, train_stats)
        normed_test_data = self._normalize(test_dataset, train_stats)

        self.TRAINING_SET = (normed_train_data, train_labels)
        self.TESTING_SET = (normed_test_data, test_labels)

    def _normalize(self, x, train_stats):
        return (x - train_stats['mean']) / train_stats['std']

    def normalizedTrainingSet(self): 
        return self.TRAINING_SET
        
    def normalizedTestingSet(self): 
        return self.TESTING_SET
