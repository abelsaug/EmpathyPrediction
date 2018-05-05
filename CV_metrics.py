from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
import numpy as np

def get_cv_metrics(text_clf, train_data, train_class, k_split):
    accuracy_scores = cross_val_score(text_clf,  # steps to convert raw messages      into models
                                      train_data,  # training data
                                      train_class,  # training labels
                                      cv=k_split,  # split data randomly into 10 parts: 9 for training, 1 for scoring
                                      scoring='accuracy',  # which scoring metric?
                                      n_jobs=-1,  # -1 = use all cores = faster
                                      )
    cv_predicted = cross_val_predict(text_clf,
                                     train_data,
                                     train_class,
                                     cv=k_split)

    return np.mean(accuracy_scores), classification_report(train_class, cv_predicted)
