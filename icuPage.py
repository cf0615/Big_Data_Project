# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
from sklearn.preprocessing import StandardScaler
import mongodb_helper as mh

icu_data = pd.read_csv('C:/Users/User/.spyder-py3/icu_preprocessed.csv')

icu_data['tooltip'] = icu_data.apply(lambda row: f"{row['state']}: {row['beds_icu_total']} cases", axis=1)

def icuPage():
    st.subheader("ICU")
    st.sidebar.header('Filters')
    icu_selected_state = st.sidebar.selectbox('Select State', ['All'] + sorted(icu_data['state'].unique()))
    icu_selected_year = st.sidebar.selectbox('Select Year', ['All'] + sorted(icu_data['year'].unique()))
    
    icu_data['date'] = pd.to_datetime(icu_data['date'])
    
    if icu_selected_state == 'All':
        st.subheader("Overall ICU Beds in Malaysia")
        if icu_selected_year == 'All':
            filtered_data = icu_data
        else:
            filtered_data = icu_data[icu_data['year'] == int(icu_selected_year)]
        
        # Group data by date and sum the total ICU beds
        grouped_data = filtered_data.groupby('date', as_index=False).sum()
        
        icu_chart = alt.Chart(grouped_data).mark_line().encode(
            x='date:T',
            y='beds_icu_total:Q',
            tooltip=['date:T', 'beds_icu_total:Q']
        ).properties(
            width=700,
            height=400
        )
        st.altair_chart(icu_chart)
        
        # Display ICU usage over time
        st.subheader('ICU Usage Over Time')
        icu_usage_chart = alt.Chart(icu_data).mark_line().encode(
            x='date:T',
            y='icu_covid:Q',
            color='state:N',
            tooltip=['date:T', 'state:N', 'icu_covid:Q']
        ).properties(
            width=700,
            height=400
        )
        st.altair_chart(icu_usage_chart)
        
        icuPredict()

        
        
    else:
        # Display data for the selected state
        st.subheader(f"Displaying Total ICU beds for {icu_selected_state}")
        icu_state_data = icu_data[icu_data['state'] == icu_selected_state]
        
        if icu_selected_year != 'All':
            icu_state_data = icu_state_data[icu_state_data['year'] == int(icu_selected_year)]
        
        #new case of selected state
        st.subheader(f"Beds for {icu_selected_state}")
        icu_state_chart = alt.Chart(icu_state_data).mark_line().encode(
                x = 'date:T',
                y = 'beds_icu_total:Q',
                tooltip=['date:T', 'beds_icu_total:Q']
            ).properties(
                    width=700,
                    height=400
                )
        st.altair_chart(icu_state_chart)
        
        # Display data for the selected state
        st.subheader(f"Displaying Total ICU beds for covid on {icu_selected_state}")
        icu_covid_beds_data = icu_data[icu_data['state'] == icu_selected_state]
        
        if icu_selected_year != 'All':
            icu_covid_beds_data = icu_covid_beds_data[icu_covid_beds_data['year'] == int(icu_selected_year)]
        
        #new case of selected state
        st.subheader(f"Beds ICU Covid for {icu_selected_state}")
        icu_covid_beds_chart = alt.Chart(icu_covid_beds_data).mark_line().encode(
                x = 'date:T',
                y = 'beds_icu_covid:Q',
                tooltip=['date:T', 'beds_icu_covid:Q']
            ).properties(
                    width=700,
                    height=400
                )
        st.altair_chart(icu_covid_beds_chart)
        
        # Display data for the selected state
        st.subheader(f"Displaying Total positive covid-19 admitted ICU for {icu_selected_state}")
        icu_covid_data = icu_data[icu_data['state'] == icu_selected_state]
        
        if icu_selected_year != 'All':
            icu_covid_data = icu_covid_data[icu_covid_data['year'] == int(icu_selected_year)]
        
        #new case of selected state
        st.subheader(f"Beds ICU Covid for {icu_selected_state}")
        icu_covid_chart = alt.Chart(icu_covid_data).mark_line().encode(
                x = 'date:T',
                y = 'icu_covid:Q',
                tooltip=['date:T', 'icu_covid:Q']
            ).properties(
                    width=700,
                    height=400
                )
        st.altair_chart(icu_covid_chart)
        
        # Comparison of icu_covid and beds_icu_covid for selected state
        st.subheader(f"Comparison of ICU COVID Admissions and COVID ICU Beds for {icu_selected_state}")
        
        comparison_chart = alt.Chart(icu_state_data).transform_fold(
            ['icu_covid', 'beds_icu_covid'],
            as_=['Category', 'Value']
        ).mark_line().encode(
            x='date:T',
            y='Value:Q',
            color=alt.Color('Category:N', scale=alt.Scale(domain=['icu_covid', 'beds_icu_covid'], range=['blue', 'red']), legend=alt.Legend(title="Category")),
            tooltip=['date:T', 'Category:N', 'Value:Q']
        ).properties(
            width=700,
            height=400
        ).interactive()
        
        st.altair_chart(comparison_chart)


def icuPredict():
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

