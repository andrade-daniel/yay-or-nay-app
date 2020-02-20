import os
import joblib
import pickle
from time import strftime
import category_encoders
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import (
    train_test_split,
    KFold,
    StratifiedKFold,
    RandomizedSearchCV,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
from sklearn import preprocessing
from utils import *

DATA_FOLDER = 'data'
OUTPUT_FOLDER = "artifacts"

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)


# Function to process data and train the model

def train_model(retrain_new_data=False):

    """
    Function that is called from the main.py 

    - It trains the model from the original raw training data or retrains with new additional data
    - Creates a pipeline for training and validation
    - Prints and saves the results on the validation data
    - Saves the trained models
    """

    if retrain_new_data == False:

        # Training only the original raw data

        # Loading data

        print("Loading original data...")
        train = pd.read_csv(DATA_FOLDER + "/training.csv", delimiter=";")
    else:
        
        # Training with old training data updated with new data

        # Loads the updated validation file with the new data
        print("Loading updated data...")
        train = pd.read_csv(OUTPUT_FOLDER + "/training_upd.csv", delimiter=";")

    # loading validation data set
    valid = pd.read_csv(DATA_FOLDER + "/validation.csv", delimiter=";")

    print(f"Shape of training data frame is: {train.shape}")
    print(f"Shape of validation data frame is: {valid.shape}")

    # Define features

    label = "classLabel"
    cols_to_drop = ["v14", "v18", "v19", label] # columns to drop from the main dataset, plus the label that is not considered in the preprocessing pipeline
    numeric_cols = ["v2", "v3", "v8", "v11", "v14", "v15", "v17", "v19"]
    categorical_cols = list(train.select_dtypes(include=["O"]).columns)

    numeric_features = [col for col in numeric_cols if col not in cols_to_drop]
    categorical_features = [
        col
        for col in categorical_cols
        if col not in cols_to_drop and col not in numeric_features
    ]


    # Create the preprocessing pipeline 

    numeric_transformer = Pipeline(
        steps=[("scaler", StandardScaler()),
            ("imputer", SimpleImputer(strategy="median"))]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("one_hot", category_encoders.OneHotEncoder(handle_unknown="impute")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    preprocessing = Pipeline(
        [
            ("col_transf", ColumnTransf()),  # cleans cols
            ("remove_cols", DropCols(cols_to_drop)),  # removes cols
            ("remove_dups", DataFrameTransf()),  # removes dups
            ("preprocessor", preprocessor),
        ]
    )

    # Fit the processing pipeline

    preprocessing.fit(train)
    
    # Save preprocessing pipeline
    joblib.dump(preprocessing, OUTPUT_FOLDER + '/preproc_pipeline.pickle')

    # Do the transformations on the datasets

    training_preproc = preprocessing.fit_transform(train)
    validation_preproc = preprocessing.transform(valid)
    # save both datasets
    joblib.dump(training_preproc, OUTPUT_FOLDER + '/training_preproc.pickle')
    joblib.dump(validation_preproc, OUTPUT_FOLDER + '/validation_preproc.pickle')
    
    print(training_preproc.shape)
    print(validation_preproc.shape)

    # Splitting datasets for training and testing

    # preprocessing the label
    train_no_dups = train.drop_duplicates()
    train_label = train_no_dups["classLabel"]
    joblib.dump(training_preproc, OUTPUT_FOLDER + '/training_label_preproc.pickle')

    le = LabelEncoder()
    le.fit(train_label)
    train_label = le.transform(train_label)
    # sabe the label encoder
    joblib.dump(le, OUTPUT_FOLDER + '/label_encoder.pickle')

    X_train, X_test, y_train, y_test = train_test_split(
        training_preproc, train_label, test_size=0.20, random_state=42
    )

    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    # Classifier

    model = RandomForestClassifier(random_state=42)

    # Fit the model

    model.fit(X_train, y_train)

    run_datetime = strftime("%m_%d_%Y_%H_%M_%S")
    
    # Save the fitted model
    joblib.dump(model, OUTPUT_FOLDER + f'/model_{run_datetime}.pickle')
    
    # Results

    # Validate on test set
    print("Preparing for validation on test set...")
    predicted = model.predict(X_test)
    print("Results almost showing...")

    # Print the results

    print("Results: \nAccuracy:", round(accuracy_score(y_test, predicted), 2))
    print("F1-score:", round(f1_score(y_test, predicted, average="macro"), 2))
    print("Precision:", round(precision_score(y_test, predicted, average="macro"), 2))
    print("Recall:", round(recall_score(y_test, predicted, average="macro"), 2))

    # Validation on validation dataset

    valid_no_dups = valid.drop_duplicates()
    valid_label = valid_no_dups.pop("classLabel")
    joblib.dump(valid_label, OUTPUT_FOLDER + '/validation_label_preproc.pickle')

    le = LabelEncoder()
    le.fit(valid_label)
    y_valid_decod = le.transform(valid_label)

    print("Preparing for validation on test set...")
    valid_preds = model.predict(validation_preproc)
    print("Results almost showing...")

    # Results on validation dataset

    acc = round(accuracy_score(y_valid_decod, valid_preds), 2)
    f1 = round(f1_score(y_valid_decod, valid_preds, average="macro"), 2)
    precision = round(precision_score(y_valid_decod, valid_preds, average="macro"), 2)
    recall = round(recall_score(y_valid_decod, valid_preds, average="macro"), 2)

    print("Results: \nAccuracy:", acc)
    print("F1-score:", f1)
    print("Precision:", precision)
    print("Recall:", recall)

    # Save the results on validation dataset
    
    model_elem = {
    "model_name": f"model_{run_datetime}",
    "Accuracy": acc,
    "F1-score": f1,
    "Precision": precision,
    "Recall": recall,
    }

    # save metric results
    pd.DataFrame(model_elem, index=[0]).to_csv('artifacts/model_valid_metrics.csv',
    mode='a', header=False, index=None)


if __name__ == '__main__':
    train_model()
