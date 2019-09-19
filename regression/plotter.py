import matplotlib.pyplot as plt
import pandas as pd

class ModelMeanErrorPlotter():
    def __init__(self, history): 
        self._history = pd.DataFrame(history.history)
        self._history['epoch'] = history.epoch
        
    def showAbsoluteErrorThroughEpochs(self):
        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Mean Abs Error [MPG]')
        plt.plot(self._history['epoch'], self._history['mean_absolute_error'], label='Train Error')
        plt.plot(self._history['epoch'], self._history['val_mean_absolute_error'], label = 'Val Error')
        plt.ylim([0,5])
        plt.legend()

    def showSquareErrorThroughEpochs(self):
        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Mean Square Error [$MPG^2$]')
        plt.plot(self._history['epoch'], self._history['mean_squared_error'], label='Train Error')
        plt.plot(self._history['epoch'], self._history['val_mean_squared_error'], label = 'Val Error')
        plt.ylim([0,20])
        plt.legend()
        plt.show()


class PredictionsPlotter:
    def __init__(self, test_labels, test_predictions):
        self._test_labels = test_labels
        self._test_predictions = test_predictions
    
    def plotTrueValuesVersusPredictions(self):
        plt.scatter(self._test_labels, self._test_predictions)
        plt.xlabel('True Values [MPG]')
        plt.ylabel('Predictions [MPG]')
        plt.axis('equal')
        plt.axis('square')
        plt.xlim([0,plt.xlim()[1]])
        plt.ylim([0,plt.ylim()[1]])
        plt.plot([-100, 100], [-100, 100])
        plt.show()

    def plotErrorDistribution(self):
        error = self._test_predictions - self._test_labels
        plt.hist(error, bins = 25)
        plt.xlabel("Prediction Error [MPG]")
        plt.ylabel("Count")
        plt.show()