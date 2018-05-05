from sklearn.externals import joblib
from sklearn import linear_model
from sklearn.feature_selection import RFECV
from CV_metrics import get_cv_metrics
import pandas as pd
from sklearn.utils import shuffle


def train_SGD():

    data = pd.read_csv("processed_data.csv", sep=',')
    data = shuffle(data)
    data_X = data.drop('Empathy', axis=1)
    data_Y = data['Empathy']

    '''TRAINING'''

    clf = linear_model.SGDClassifier(loss='hinge', penalty='l1',alpha=1e-4, random_state=16,max_iter=10000, tol=0.0001, shuffle=True)    
    clf = RFECV(clf, step=1, cv=5, n_jobs=-1).fit(data_X,data_Y.values.ravel())
    joblib.dump(clf, 'SGD_model.pkl')

    
    """PERFORMANCE EVALUATION"""
    accuracy, clf_report = get_cv_metrics(clf, data_X, data_Y.values.ravel(), k_split=10)
    print("Accuracy: ", accuracy)
    print(clf_report)
    return clf
    

if __name__ == "__main__":
    train_SGD()
