from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.model_selection import KFold
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
import pandas as pd
from sklearn.utils import shuffle

if __name__ == "__main__":

    data = pd.read_csv("processed_data.csv", sep=',')
    data = shuffle(data)
    data_X = data.drop('Empathy', axis=1)
    data_Y = data['Empathy']

    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(data_Y)
    encoded_Y = encoder.transform(data_Y)
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y = np_utils.to_categorical(encoded_Y)

    def save_model(model):
        # saving model
        json_model = model.to_json()
        open('model_architecture.json', 'w').write(json_model)
        # saving weights
        model.save_weights('model_weights.h5', overwrite=True)

    # define baseline model
    def nn_model():
        # create model
        model = Sequential()
        model.add(Dense(70, input_dim=150, activation='relu'))
        model.add(Dense(40, activation='relu'))
        model.add(Dense(5, activation='softmax'))
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model



    estimator = nn_model()
    # KerasClassifier(build_fn=nn_model, epochs=100, batch_size=512, verbose=0)
    estimator.fit(data_X, dummy_y, epochs=100, batch_size=512, verbose=0)
    save_model(estimator)

    #Model Evaluation
    # """PERFORMANCE EVALUATION"""
    # seed = 7
    # np.random.seed(seed)
    # kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
    # results = cross_val_score(estimator, data_X, dummy_y, cv=kfold)
    # print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

