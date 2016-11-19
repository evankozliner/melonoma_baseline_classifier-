import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from logistic_regression_classifier import plot_confusion_matrix

if __name__ == "__main__":
    m = np.loadtxt('out.csv', dtype=(int,int), delimiter= ",")
    predicted = m[:,0]
    actual = m[:,1]
    plt.figure
    #plot_confusion_matrix(confusion_matrix(actual, predicted), list(set(actual)))
    plt.show()

    print str((np.sum(np.equal(predicted,actual)) / float(m.shape[0])))
