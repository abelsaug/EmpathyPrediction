from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from sklearn.feature_selection import RFECV
from CV_metrics import get_cv_metrics
import pandas as pd
from sklearn.utils import shuffle

def train_LR():
    
    data = pd.read_csv("processed_data.csv", sep=',')
    data = shuffle(data)
    data_X = data.drop('Empathy', axis=1)
    data_Y = data['Empathy']
    
    '''TRAINING'''
    clf = LogisticRegression(C = 2.6, dual=False, penalty = 'l2', tol = 1e-4, solver = 'saga', n_jobs = -1, multi_class = 'ovr', fit_intercept = True, intercept_scaling = 1.4).fit(data_X,data_Y.values.ravel())
##    clf = RFECV(clf, step=1, cv=5, n_jobs=-1).fit(data_X,data_Y)
    joblib.dump(clf, 'LR_model.pkl')

    """PERFORMANCE EVALUATION"""
    accuracy, clf_report = get_cv_metrics(clf, data_X, data_Y.values.ravel(), k_split=10)
    print("Accuracy: ", accuracy)
    print(clf_report)
    return clf

if __name__ == "__main__":
    train_LR()
