from imdb_dataset_provider import * 
from basic_text_classifier import * 
from plotter import ModelAccuracyPlotter 

# Example from https://www.tensorflow.org/tutorials/keras/basic_text_classification

imdbDataset = IMDBDatasetProvider(words_in_training_data=10000)
(train_data, train_labels) = imdbDataset.paddedTrainingSet()
(test_data, test_labels) = imdbDataset.paddedTestingSet()

classifier = BasicTextClassifier()

history = classifier.train(
    normalized_training_data=train_data,
    training_label=train_labels, 
    validate_on_first=10000, 
    epochs=40, 
    batch_size=512)

results = classifier.evaluateModelAccuracy(
    normalized_testing_data=test_data, 
    testing_labels=test_labels)

print(results)

plotter = ModelAccuracyPlotter(history)
plotter.showLossThroughEpochs()
plotter.showAccuracyThroughEpochs()