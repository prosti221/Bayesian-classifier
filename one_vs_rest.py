import numpy as np
import scipy as sp
import sklearn.metrics as skm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Combines multiple gaussian models and predicts the class from the most confident
class oneVsRest():
    def __init__(self, sub_classifiers):
        self.sub_classifiers = sub_classifiers
        self.num_class = sub_classifiers[0].class_num
        self.dims = sub_classifiers[0].y.shape
        self.accuracy = 0
        self.cm = 0 # Confusion matrix
        self.predictions = 0
        self.y = sub_classifiers[0].y

    def predict(self):
        preds = np.zeros(self.dims)
        for i in range(self.dims[0]):
            for j in range(self.dims[0]):
                probs = [k.probabilities[i,j] for k in self.sub_classifiers]
                most_confident = np.argmax([max(i) for i in probs])
                preds[i, j] = np.argmax(probs[most_confident])
        self.get_accuracy(preds, self.y)
        self.get_confusion_mat(preds, self.y)
        return preds

    def get_accuracy(self, y_pred, y):
        self.accuracy = np.mean(y_pred[y!=0] == y[y!=0]) #accuracy excluding the zero label
        return self.accuracy

    def get_confusion_mat(self, y_pred, y):
        self.cm = skm.confusion_matrix(y[y!=0].flatten(), y_pred[y!=0].flatten())
        return self.cm

    def plot_confusion_mat(self):
        if self.cm.shape == (4, 4):
            rows = [i for i in range(1, self.cm.shape[0] + 1)]
            col = [i for i in range(1, self.cm.shape[1] + 1)]
        else:
            rows = [i for i in range(self.cm.shape[0])]
            col = [i for i in range(self.cm.shape[1])]

        cm_df = pd.DataFrame(self.cm, index = rows, columns = col)

        plt.figure(figsize=(5,5))
        sns.heatmap(cm_df, annot=True, fmt='g')
        plt.title('Confusion Matrix')
        plt.ylabel('Actal Values')
        plt.xlabel('Predicted Values')
        plt.show()
