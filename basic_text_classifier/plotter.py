import matplotlib.pyplot as plt

class ModelAccuracyPlotter:
    def __init__(self, history):
        self._history = history.history

    def showLossThroughEpochs(self):
        plt.clf()   # clear figure

        loss = self._history['loss']
        val_loss = self._history['val_loss']
        epochs = range(1, len(loss) + 1)

        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.show()

    def showAccuracyThroughEpochs(self):
        plt.clf()   # clear figure

        acc = self._history['acc']
        val_acc = self._history['val_acc']
        epochs = range(1, len(acc) + 1)

        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.show()