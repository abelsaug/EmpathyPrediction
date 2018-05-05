from sklearn.externals import joblib
from keras.models import Sequential, model_from_json
import pandas as pd
from CV_metrics import get_cv_metrics
from sklearn.utils import shuffle
from sklearn import preprocessing

if __name__ == "__main__":
    test_data = pd.read_csv("test_processed_data.csv", sep=',')
    test_data = shuffle(test_data)
    test_data_X = test_data.drop('Empathy', axis=1)
    test_data_Y = test_data['Empathy']



    df_X = pd.DataFrame()
    cols = list(test_data_X)
    for col in cols:
        print(col)
        column = test_data_X[[col]].values.astype(float)
        # Create a minimum and maximum processor object
        min_max_scaler = preprocessing.MinMaxScaler()
        
        # Create an object to transform the data to fit minmax processor
        column_scaled = min_max_scaler.fit_transform(column)
        # Run the normalizer on the dataframe
        df_X = pd.concat([df_X, pd.DataFrame(column_scaled, columns=test_data_X[[col]].columns)], axis=1)
    test_data_X = df_X



    def load_model():
        # loading model
        model = model_from_json(open('model_architecture.json').read())
        model.load_weights('model_weights.h5')
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        return model


    clf1 = joblib.load('SGD_model.pkl')
    pred_y = clf1.predict(test_data_X)
    print("Test accuracy for Stochastic Gradient Descent Classifier: ", accuracy_score(pred_y, test_data_Y))

    clf2 = joblib.load('SVC_model.pkl')
    pred_y = clf2.predict(test_data_X)
    print("Test accuracy for Support Vector Classifier: ", accuracy_score(pred_y, test_data_Y))

    clf3 = joblib.load('RF_model.pkl')
    pred_y = clf3.predict(test_data_X)
    print("Test accuracy for Random Forest: ", accuracy_score(pred_y, test_data_Y))

    clf4 = joblib.load('Bernoulli_nb_model.pkl')
    pred_y = clf4.predict(test_data_X)
    print("Test accuracy for Bernoulli Naive Bayes: ", accuracy_score(pred_y, test_data_Y))

    clf5 = joblib.load('Multinomial_nb_model.pkl')
    pred_y = clf5.predict(test_data_X)
    print("Test accuracy for Multinomial Naive Bayes: ", accuracy_score(pred_y, test_data_Y))

    clf6 = joblib.load('LR_model.pkl')
    pred_y = clf6.predict(test_data_X)
    print("Test accuracy for Logistic Regression: ", accuracy_score(pred_y, test_data_Y))

    clf7 = joblib.load('ET_model.pkl')
    pred_y = clf7.predict(test_data_X)
    print("Test accuracy for Extra Tree Classifier: ", accuracy_score(pred_y, test_data_Y))

    clf8 = joblib.load('Voting_model.pkl')
    pred_y = clf8.predict(test_data_X)
    print("Test accuracy for Voting Classifier: ", accuracy_score(pred_y, test_data_Y))

    #neural Network model
    model = load_model()
    # predictions
    pred_y = model.predict_classes(test_data_X)
    print("Test accuracy for Neural Networks: ", accuracy_score(pred_y, test_data_Y))

