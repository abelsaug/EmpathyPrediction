from datacleaner import autoclean
import pandas as pd
from sklearn.utils import shuffle
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE
from imblearn.datasets import make_imbalance
from sklearn.decomposition import PCA

if __name__ == "__main__":

    data = pd.read_csv("responses.csv", sep=',')
    data = shuffle(data)
    data_X = data.drop('Empathy', axis=1)
    data_Y = data['Empathy']

    #Drops rows with missing target values
    for i,nullbool in enumerate(data_Y.isnull()):
        if nullbool==True:
            data_Y = data_Y.drop(data.index[i])
            data_X = data_X.drop(data.index[i])
    data_Y = data_Y.reset_index(drop=True)
    data_X = data_X.reset_index(drop=True)
    data = pd.concat([data_X, data_Y], axis=1)
    #Autoclean
    autoclean(data, drop_nans=False, copy=False, ignore_update_check=False)

    ##Split to test set

    train_data = data[:-150]
    test_data = data[-150:]
    test_data_X = test_data.drop('Empathy', axis=1)
    test_data_Y = test_data['Empathy']
    data = train_data
    data_Y = data['Empathy']
    data_X = data.drop('Empathy', axis=1)

    # One hot encoding
    # data_X_O = data_X.drop('Height', axis=1)
    # data_X_O = data_X_O.drop('Weight', axis=1)
    # data_X_O = data_X_O.drop('Age', axis=1)


    # cols = list(data_X_O)

    # encoder = OneHotEncoder(cols)
    # encoder.fit(data_X_O)
    # data_X_O = encoder.transform(data_X_O)
    # Height = data_X['Height']
    # Weight = data_X['Weight']
    # Age = data_X['Age']
    # data_X = pd.concat([data_X_O, Height, Weight, Age], axis=1)

    # Normalization of columns
    df_X = pd.DataFrame()
    cols = list(data_X)
    for col in cols:
        column = data_X[[col]].values.astype(float)
        # Create a minimum and maximum processor object
        min_max_scaler = preprocessing.MinMaxScaler()
        
        # Create an object to transform the data to fit minmax processor
        column_scaled = min_max_scaler.fit_transform(column)
        # Run the normalizer on the dataframe
        df_X = pd.concat([df_X, pd.DataFrame(column_scaled, columns=data_X[[col]].columns)], axis=1)
    data_X = df_X

    cols = list(data_X.columns)
    colsY = ['Empathy']

    smote = SMOTE(kind = "borderline1", k_neighbors  = 10)
    data_X_without_headers, data_Y_without_headers = smote.fit_sample(data_X, data_Y)

    data_X_without_SMOTE = data_X
    data_Y_without_SMOTE = data_Y

    data_X =  pd.DataFrame(data_X_without_headers, columns=cols)
    data_Y =  pd.DataFrame(data_Y_without_headers, columns=colsY)

    ##pca = PCA(0.95)
    ##pca.fit(data_X)
    ##print(pca.explained_variance_)
    ##pca.n_components_
    ##data_X_new = pca.transform(data_X)
    ##test_data_X_new = pca.transform(test_data_X)
    ##data_new = pd.concat([pd.DataFrame(data_X_new), data_Y], axis=1)

    data = pd.concat([pd.DataFrame(data_X), data_Y], axis=1)
    data.to_csv('processed_data.csv')
    test_data.to_csv('test_processed_data.csv')

