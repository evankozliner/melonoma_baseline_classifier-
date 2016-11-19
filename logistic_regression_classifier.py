from __future__ import division

from sklearn import linear_model, datasets
from sklearn.metrics import confusion_matrix
from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np

NUM_TEST_SAMPLES = 5000
NUM_TRAIN_SAMPLES = 5000
EXT = '.pickle'

def shuffle_images_and_labels(images,labels):
    rng_state = np.random.get_state()
    np.random.shuffle(images)
    np.random.set_state(rng_state)
    np.random.shuffle(labels)
    images = images.reshape(images.shape[0], images.shape[1] * images.shape[2] * images.shape[3])
    labels = labels.reshape(labels.shape[0])
    return images, labels

def pluck_data(dataset):
    data_ben = np.load(dataset + '/' + 'Benign' + EXT)
    data_mal = np.load(dataset + '/' + 'Malignant' + EXT)
    Y_mal = np.ones(data_mal.shape[0]).reshape(data_mal.shape[0],1)
    Y_ben = np.zeros(data_ben.shape[0]).reshape(data_ben.shape[0],1)
    images = np.vstack([data_ben, data_mal])
    labels = np.vstack([Y_ben, Y_mal])
    return shuffle_images_and_labels(images,labels)

def plot_confusion_matrix(cm, labels,title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def print_confusion_scores(conf):
    tp = conf[1,1]
    tn = conf[0,0]
    fp = conf[0,1]
    fn = conf[1,0]
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    recall_sens = tp / (tp + fn)
    precision = tp / (tp + fp)
    specificity = tn / (tn + fp)
    fp_rate = fp / (fp + tp)
    fn_rate = fn / (fn + tn)
    print "Accuracy : " + str(accuracy)
    print "Recall / Sensitvity : " + str(recall_sens)
    print "Precision : " + str(precision)
    print "Specificity : " +str(specificity)
    print "False Positive Rate : " + str(fp_rate)
    print "False Negative Rate : " + str(fn_rate)

def main():
    print "Getting data..."
    X, Y = pluck_data('train')
    X_test, Y_test = pluck_data('test')
    print "Finished getting data. Training model..."
    clf = svm.SVC()
    print X.shape
    print Y.shape
    clf.fit(X[:NUM_TRAIN_SAMPLES,:], Y[:NUM_TRAIN_SAMPLES])
    Y_predict = clf.predict(X_test[:NUM_TEST_SAMPLES,:])
    plt.figure
    conf_mat = confusion_matrix(Y_test[:NUM_TEST_SAMPLES], Y_predict[:NUM_TEST_SAMPLES])
    print_confusion_scores(conf_mat)
    print conf_mat
    plot_confusion_matrix(conf_mat, list(set(Y_test[:NUM_TEST_SAMPLES])))
    plt.show()

if __name__ == "__main__":
    main()
