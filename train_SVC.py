from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.feature_selection import RFECV
from CV_metrics import get_cv_metrics
import pandas as pd
from sklearn.utils import shuffle


def train_SVM():
    
    data = pd.read_csv("processed_data.csv", sep=',')
    data = shuffle(data)
    data_X = data.drop('Empathy', axis=1)
    data_Y = data['Empathy']
   
    '''TRAINING'''
    clf = SVC(C = 3.1, kernel='rbf', gamma = 0.1, tol = 1e-1).fit(data_X,data_Y.values.ravel())
    joblib.dump(clf, 'SVC_model.pkl')

    """PERFORMANCE EVALUATION"""
    accuracy, clf_report = get_cv_metrics(clf, data_X, data_Y.values.ravel(), k_split=10)
    print("Accuracy: ", accuracy)
    print(clf_report)
    return clf


if __name__ == "__main__":
    train_SVM()

