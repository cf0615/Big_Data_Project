# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import mongodb_helper as mh

def icuPage():
    # Load the trained model, scaler, and feature columns
    model = mh.load_model_from_db('xgb_icu_prediction.pkl', 'predict_icu_xgb')
    scaler = mh.load_model_from_db('scaler.pkl', 'predict_icu_xgb')
    feature_columns = mh.load_model_from_db('feature_columns.pkl', 'predict_icu_xgb')

    # Streamlit app
    st.title('ICU Admission Prediction')

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Read the uploaded CSV file
        input_df = pd.read_csv(uploaded_file)

        # Display the uploaded data
        st.subheader('Uploaded Data')
        st.write(input_df.head())  # Display the first few rows of the data

        # Display the shape of the data
        st.subheader('Shape of the Data')
        st.write(input_df.shape)

        # Display the columns of the data
        st.subheader('Columns in the Data')
        st.write(input_df.columns.tolist())

        # Remove columns that should not be part of the prediction input
        if 'ICU' in input_df.columns:
            input_df = input_df.drop(columns=['ICU'])


        # Recheck the columns of the data after dropping unnecessary columns
        st.subheader('Columns after Dropping Unnecessary Columns')
        st.write(input_df.columns.tolist())

        # Check if there are any missing or extra columns
        missing_columns = [col for col in feature_columns if col not in input_df.columns]
        extra_columns = [col for col in input_df.columns if col not in feature_columns]

        if missing_columns:
            st.error(f"The uploaded file is missing the following columns: {', '.join(missing_columns)}")
        if extra_columns:
            st.error(f"The uploaded file has extra columns that were not used during training: {', '.join(extra_columns)}")

        # Proceed if there are no missing or extra columns
        if not missing_columns and not extra_columns:
            try:
                # Preprocess input data
                input_scaled = scaler.transform(input_df[feature_columns])

                # Make prediction
                prediction = model.predict(input_scaled)
                prediction_prob = model.predict_proba(input_scaled)

                # Display the predictions
                st.subheader('Prediction')
                input_df['ICU Admission Prediction'] = prediction
                st.write(input_df[['ICU Admission Prediction']])

                st.subheader('Prediction Probability')
                st.write(pd.DataFrame(prediction_prob, columns=[f'Probability of class {i}' for i in range(prediction_prob.shape[1])]))
            except ValueError as e:
                st.error(f"Error in scaling or prediction: {e}")
                st.write(f"Expected {len(feature_columns)} features, but got {input_df.shape[1]} features.")
    else:
        st.info("Please upload a CSV file to proceed.")
