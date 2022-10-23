import numpy as np
import scipy as sp
import sklearn.metrics as skm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class bayesianModel():
    def __init__(self):
        self.mean_vecs = [] # Mean values for each class
        self.cov_mats = []  # Standard deviation for each class
        self.c_priors = [0.1e-7, 0.25-(0.1e-7/4),  0.25-(0.1e-7/4),  
                         0.25-(0.1e-7/4),  0.25-(0.1e-7/4)]   # Class prior probability
        self.class_num = 0  # Number of classes
        self.accuracy = 0   # Accuracy from last training
        self.cm = 0         # Confusion matrix
        self.probabilities = 0
        
        self.x = None       # Last training input
        self.y = None       # Last training true values
        self.preds = None

    # Gaussian distribution P(X | Y) * P(Y)
    def gauss(self, X, mean_vec, cov_mat, prior):
        k = X.shape[0]; m = np.matrix(X-mean_vec)
        return ((1/( (2*np.pi)**0.5*sp.linalg.det(cov_mat)**0.5)) \
                * np.exp(-0.5*(X-mean_vec)@sp.linalg.inv(cov_mat)@(X-mean_vec).T)) * prior
        
    def softmax(self, x):
        return x/sum(x)
    
    # Predict for each pixel
    def predict(self, x):
        self.probabilities = np.zeros((x.shape[0], x.shape[1], self.class_num))
        preds = np.zeros((x.shape[0], x.shape[1])).astype(int)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                norm_vals = ([self.gauss(x[i, j], self.mean_vecs[c], self.cov_mats[c], 
                                         self.c_priors[c]) for c in range(self.class_num)])
                preds[i, j] = np.argmax(norm_vals)
                self.probabilities[i, j] = norm_vals
                #print(self.softmax(norm_vals))
        self.get_accuracy(preds, self.y)
        self.get_confusion_mat(preds, self.y)
        self.preds = preds
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

    # Train the classifier
    def fit(self, x, y):
        self.x = x; self.y = y
        self.class_num = len(np.unique(y))
        #Mean-value vector for each feature f calcualted for each class c
        self.mean_vecs = [np.mean(x[y==i], axis=(0, 1)) for i in range(self.class_num)]
        #Covariance matrix from all features f calculated for each class c
        self.cov_mats = [np.cov(x[y==i], rowvar=False) for i in range(self.class_num)]
        #self.c_priors = np.array([y[y==i].shape[0] for i in range(self.class_num)]) / (y[y!=0].flatten().shape[0])
