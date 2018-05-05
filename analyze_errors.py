from sklearn.externals import joblib

clf3 = joblib.load('SGD_model.pkl')
pred_y = clf3.predict(test_data_X)
for true, pred in zip(pred_y, test_data_Y):
    if pred != true:
        print(test_data_X.iloc[[index]],pred,true)
