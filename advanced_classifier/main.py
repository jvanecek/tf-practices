
from imdb_dataset_provider import IMDBDatasetProvider
from advanced_classifiers import BaselineClassifier, SmallerModelClassifier, BiggerModelClassifier, L2RegularizedClassifier, DropoutRegularizedClassifier
from plotter import BinaryCrossEntropyPlotter

imdbDataset = IMDBDatasetProvider(words_in_training_data=10000)
(train_data, train_labels) = imdbDataset.multiHotTrainingSet()
(test_data, test_labels) = imdbDataset.multiHotTestingSet()

baseline_history = BaselineClassifier().train(train_data, train_labels, validation_data=(test_data, test_labels), input_size=10000)
smaller_history  = SmallerModelClassifier().train(train_data, train_labels, validation_data=(test_data, test_labels), input_size=10000)
bigger_history = BiggerModelClassifier().train(train_data, train_labels, validation_data=(test_data, test_labels), input_size=10000)
l2_history = L2RegularizedClassifier().train(train_data, train_labels, validation_data=(test_data, test_labels), input_size=10000)
dpt_history = DropoutRegularizedClassifier().train(train_data, train_labels, validation_data=(test_data, test_labels), input_size=10000)

BinaryCrossEntropyPlotter().showHistoryThroughEpochs([
    ('baseline', baseline_history),
    ('smaller', smaller_history),
    ('bigger', bigger_history)])

BinaryCrossEntropyPlotter().showHistoryThroughEpochs([
    ('baseline', baseline_history),
    ('l2', l2_history)])

BinaryCrossEntropyPlotter().showHistoryThroughEpochs([
    ('baseline', baseline_history),
    ('dropout', dpt_history)])