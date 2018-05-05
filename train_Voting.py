from sklearn.ensemble import VotingClassifier
from sklearn.externals import joblib
import pandas as pd
from sklearn.utils import shuffle
from CV_metrics import get_cv_metrics

def train_VotingClassifier():

    data = pd.read_csv("processed_data.csv", sep=',')
    data = shuffle(data)
    data_X = data.drop('Empathy', axis=1)
    data_Y = data['Empathy']
    
    """TRAINING"""

    clf1 = joblib.load('SGD_model.pkl')
    clf2 = joblib.load('SVC_model.pkl')
    clf3 = joblib.load('RF_model.pkl')
#     clf4 = joblib.load('Bernoulli_nb_model.pkl')
#     clf5 = joblib.load('Multinomial_nb_model.pkl')
    clf6 = joblib.load('LR_model.pkl')
    clf7 = joblib.load('ET_model.pkl')

    eclf1 = VotingClassifier(estimators=[('SGD', clf1),('SVC', clf2), ('RF', clf3), ('ET', clf7)], voting='hard')
    eclf1.fit(data_X, data_Y)
    joblib.dump(eclf1, 'Voting_model.pkl')
    
    """PERFORMANCE EVALUATION"""

    accuracy, clf_report = get_cv_metrics(eclf1, data_X, data_Y, k_split=10)
    print("Accuracy: ", accuracy) #Accuracy: 73.06
    print(clf_report)

    return eclf1


if __name__ == "__main__":
    train_VotingClassifier()

