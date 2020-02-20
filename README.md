# Machine Learning + Microservice CLI & Deployed App in Heroku (with some analysis over Jupyter Notebook)

## Goal

It is a simple binary classification exercise, where the goal is to:

1. Create and train a machine learning model using the training set that performs well on the validation set.

2. To create a microservice that will serve predictions of that model.

3. To build it so that it learns online (meaning it accepts labeled data one by one and gradually improves the predictions) using the given dataset as initial "starter".

For that, you can use it in different ways:
* Either by using the command line (instructions below)
* Or by using the deployed app, that you can find here (to be used with the files in this repo): 

    https://yay-or-nay-app.herokuapp.com.

You can also find a Jupyter Notebook with some exploration of the data, pipelining, modelling and explained choices.


### Pre-requisites

Having Python 3 installed (Anaconda is recommended).

Or, if using just the Heroku app, at least to leave the files/folders of this repo.

### Installing

At the command prompt:

```
> pip install -r requirements.txt
```

### Getting started

* To run the Jupyter notebook (following, it will open in your browser, from where you'll be able to open the notebook; also available here, is a html version of it for quick access):

```
> jupyter notebook
```


* If you want to use the command line, you can use the main.py file with some arguments that will call the other scripts (train.py and predict.py).

You can run it and use arguments to train the model, to retrain it with new added data or to run a prediction. Some examples:

    - To train a model:

```
    > python main.py -train
```

    - To make a prediction on new data:

```
    > python main.py -predict -file_path=\artifacts\example_to_predict.json -file_type=json -model=\artifacts\model_02_20_2020_14_06_43.pickle
```
Where you have arguments to specify that it's a prediction, the path of the (json or csv) file having the new observation and the path of the model to use.

    - To retrain a model, after having added new data:

```
    > python main.py -retrain
```


* Regarding the app, the purpose is to run a prediction or to submit new labeled data to improve our classifier. To use it, you have two options.

To use the link above or to run it locally:

```
    > streamlit run app.py
```
In either case, you will need the files in the artifacts folder.

There you can find two examples of new observations, one as a json file, the other as a csv file. They serve as a demonstration of the usage of new data to use our classifier and/or to show how to add new data for retraining. They can be upload directly in the app!

Every time you train or retrain a model, the results on the validation dataset will be added in a csv file (model_valid_metrics.csv). Other files are saved too (pipelines, label encoders, etc.).

Every time you add a new observation, it will be added in a csv file that will serve as the training data (adding to the original training data) when retraining is needed (training_upd.csv).

Every time you (re)train a new model, it will be saved there (model.pickle or model[datetime].pickle). These models will be available for uploading in the app, so you can choose wich one to use there too!


Have fun!!!
