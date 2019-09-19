
from autompg_dataset_provider import AutoMPGDatasetProvider
from regression_predictor import RegressionPredictor
from utils import * 
from plotter import ModelMeanErrorPlotter, PredictionsPlotter

# Example from https://www.tensorflow.org/tutorials/keras/basic_regression

datasetProvider = AutoMPGDatasetProvider()
(normed_train_data, train_labels) = datasetProvider.TRAINING_SET
(normed_test_data, test_labels) = datasetProvider.TESTING_SET

predictor = RegressionPredictor()

print("\nTrain all 1000 epochs")
history = predictor.train( 
    normalized_training_data=normed_train_data, 
    training_labels=train_labels, 
    epochs=1000, 
    callbacks=[DotPrinter()])

plotter = ModelMeanErrorPlotter(history)
plotter.showAbsoluteErrorThroughEpochs()
plotter.showSquareErrorThroughEpochs()

loss, meanAbsError, meanSquareError = predictor.evaluateModelAccuracy(normed_test_data, test_labels)
print("\nTesting set Mean Abs Error: {:5.2f} MPG".format(meanAbsError))

print("\nTrain 1000 epochs with early stop")
history = predictor.train( 
    normalized_training_data=normed_train_data, 
    training_labels=train_labels, 
    epochs=1000, 
    callbacks=[early_stop, DotPrinter()])

plotter = ModelMeanErrorPlotter(history)
plotter.showAbsoluteErrorThroughEpochs()
plotter.showSquareErrorThroughEpochs()

loss, meanAbsError, meanSquareError = predictor.evaluateModelAccuracy(normed_test_data, test_labels)
print("\nTesting set Mean Abs Error: {:5.2f} MPG".format(meanAbsError))

test_predictions = predictor.predict(normed_test_data)

plotter = PredictionsPlotter(test_labels, test_predictions)
plotter.plotTrueValuesVersusPredictions()
plotter.plotErrorDistribution()