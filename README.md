## Empathy prediction on Young People Survey Dataset ( https://www.kaggle.com/miroslavsabo/young-people-survey/ )


### Install required libraries:
    
```

    pip install -r requirements.txt
    pip install -e 'git+http://github.com/dustinstansbury/stacked_generalization.git#egg=stacked_generalization' (OPTIONAL)


```


### Instructions to preprocess the data:

```

    python data_cleaning.py


```

### Instructions to train the model:

```

    python train_{model}.py


```

### Instructions to test the model (will use pre-trained model):

```

    python predict_model.py


```


### Instructions to run  Jupyter Notebook:

```

    jupyter notebook Predicting Empathy.ipynb


```
