import joblib
import pandas as pd
from train import *
from utils import * 

OUTPUT_FOLDER = "artifacts"

def predict_label(NEW_DATA, OUTPUT_FOLDER, MODEL=None):
    '''
    Predict function

        - params: 
                - NEW_DATA: input of the new data to predict on
                - OUTPUT_FOLDER: is the folder where files will ne saved
                - MODEL: model to be used; if None, it will use the first model trained;
                                            else, receives other model path
        - return: predicted class
    '''

    # Load preprocessing pipeline
    preproc_pipeline = joblib.load(OUTPUT_FOLDER + '/preproc_pipeline.pickle')

    # Load trained model
    if MODEL == None:
        # loads first trained model
        model = joblib.load(OUTPUT_FOLDER + '/model.pickle')
    else:
        # loads model's from path
        model = joblib.load(MODEL)

    # Load label encoder to decode the prediction
    le = joblib.load(OUTPUT_FOLDER + '/label_encoder.pickle')
    
    # Preprocess the input data
    
    NEW_DATA_PREPROC = preproc_pipeline.transform(NEW_DATA)

    # Prediction
    prediction = model.predict(NEW_DATA_PREPROC)

    # Recode the prediction
    recod_pred = le.inverse_transform(prediction)[0]

    print(f"The predicted label is: {recod_pred}")

    return recod_pred

if __name__ == '__main__':

    validation = pd.read_csv(DATA_FOLDER + "/validation.csv", delimiter=";")
    preprocessing = joblib.load(OUTPUT_FOLDER + '/preproc_pipeline.pickle')
    
    VALIDATION_EXAMPLE_PREPROC = validation.head(1)

    predict_label(VALIDATION_EXAMPLE_PREPROC, OUTPUT_FOLDER)