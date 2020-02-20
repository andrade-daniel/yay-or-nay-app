import json
import streamlit as st
from PIL import Image
from predict import *
from train import *

OUTPUT_FOLDER = "artifacts"

img = Image.open(OUTPUT_FOLDER + '/yay-nay.jpg')
st.sidebar.image(img, use_column_width=False, width=200)

st.title('YES or NO app')

st.sidebar.subheader("What to to?")

app_mode = st.sidebar.radio(label="", options=["Run Prediction", "Submit new labeled data"])

if app_mode == "Run Prediction":

    uploaded_file = st.file_uploader(label="1) Select the file with the new observation (json, csv)")
    file_types = ['json', 'csv']
    sel_file_type = st.selectbox(label="Select the file format", options=file_types, \
        index=0)

    if uploaded_file is not None:
        if st.button("Click to confirm"):
            if sel_file_type == 'json':
                data_to_predict = pd.read_json(uploaded_file)
                
            elif sel_file_type == 'csv':
                data_to_predict = pd.read_csv(uploaded_file)
            joblib.dump(data_to_predict, "artifacts/newdata.pickle")
            st.info("You may proceed!")
        
        uploaded_model = st.file_uploader(label="2) Select the model you want to use for classification or you can skip this step and proceed to the next one (it will use the first trained model)") 
        if uploaded_model is not None:
            if st.button("Click to confirm the chosen model"):
                model = joblib.load(uploaded_model)
                st.info("Ready!")
            else:
                model = None

            # Run prediction
            submit = st.button('Run the classifier!')
            if submit:
                data_to_predict = joblib.load("artifacts/newdata.pickle")
                st.write(data_to_predict)
                with st.spinner('Wait for it...'):
                    st.subheader('The predicted class is:')
                    prediction = predict_label(data_to_predict, OUTPUT_FOLDER, MODEL=model)
                    
                    if len(data_to_predict) == 0:
                        st.subheader("Please, upload a file!")
                    else:
                        st.subheader(prediction)
                        st.success('Done!')

    click = st.button("Click here when it's done")
    if click:
        with st.spinner('Wait for it...'):
            img_future = Image.open(OUTPUT_FOLDER + '/see-you-in-the-future.png')
            st.image(img_future,
            use_column_width=False, width=600)
            
else: #app_mode == "Insert New Data":
   
    uploaded_file = st.file_uploader(label="")

    file_types = ['json', 'csv']
    sel_file_type = st.selectbox(label="Select the file format", options=file_types, \
        index=0)

    if uploaded_file is not None:
        if st.button("Click to submit"):
            if sel_file_type == 'json':
                new_data = pd.read_json(uploaded_file)
                joblib.dump(new_data, "artifacts/newdata.pickle")
                st.write(new_data)
            elif sel_file_type == 'csv':
                new_data = pd.read_csv(uploaded_file)
                joblib.dump(new_data, "artifacts/newdata.pickle")
                st.write(new_data)
            st.success('Submission successful!')
            new_data.to_csv('artifacts/training_upd.csv', mode='a', header=False, index=None, sep=";")

        st.subheader("Tick the checkbox below if you want to retrain the model with new submitted data and to receive the results.")
        
        if st.checkbox("Retrain the model with the new data"):
            train_model(retrain_new_data=True)
            model_valid_metrics = pd.read_csv(OUTPUT_FOLDER + "/model_valid_metrics.csv")
            st.success('Retraining done!')

            st.subheader("All results for model comparison")
            st.write(model_valid_metrics)



   

        