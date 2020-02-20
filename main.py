import argparse
import joblib
from train import train_model
from predict import *

OUTPUT_FOLDER = "artifacts"

if __name__ == '__main__':

    # parser argument definition

    parser = argparse.ArgumentParser()

    parser.add_argument("-predict", "--predict",
    # nargs='+',
    help = "Command to make a prediction",
    required = False,
    action='store_true')

    parser.add_argument("-model", "--model",
    help = "Model to be used for prediction",
    required = False)

    parser.add_argument("-train", "--train",
    help = "Command to train the model",
    required = False,
    action='store_true')

    parser.add_argument("-retrain", "--retrain",
    help = "Command to retrain the model with new data",
    required = False,
    action='store_true')

    parser.add_argument("-file_path", "--file_path",
    help = "The file with the new observation to use the predict function on",
    required = False)

    parser.add_argument("-file_type", "--file_type",
    help = "The file type",
    required = False)

    # Retrieve the arguments
    args = parser.parse_args()
    predict = args.predict
    train = args.train
    retrain = args.retrain
    file_path = args.file_path
    file_type = args.file_type
    model= args.model

    # Train, retrain or predict, according to the commands

    # To train the model
    if train == True:
        train_model() # train the model
    
    # To retrain the model
    elif retrain == True:
        train_model(retrain_new_data=True) # retrain the model with new data

    # To make a prediction
    elif predict == True and file_path:
        
        if file_type == 'csv':
            NEW_DATA = pd.read_csv(file_path)
        elif file_type == 'json':
            NEW_DATA = pd.read_json(file_path)
        
        predict_label(NEW_DATA, OUTPUT_FOLDER, MODEL=model) # use the predict function
    
    else:
        print("Please, use the arguments correctly!")
    