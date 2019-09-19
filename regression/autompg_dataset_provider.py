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

        dataset = raw_dataset.copy().dropna()

        origin = dataset.pop('Origin')
        dataset['USA'] = (origin == 1)*1.0
        dataset['Europe'] = (origin == 2)*1.0
        dataset['Japan'] = (origin == 3)*1.0

        train_dataset = dataset.sample(frac=0.8,random_state=0)
        test_dataset = dataset.drop(train_dataset.index)

        train_stats = train_dataset.describe()
        train_stats.pop("MPG")
        train_stats = train_stats.transpose()

        train_labels = train_dataset.pop('MPG')
        test_labels = test_dataset.pop('MPG')

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
