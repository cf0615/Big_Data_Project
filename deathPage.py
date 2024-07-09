import streamlit as st
import pandas as pd
import altair as alt
import mongodb_helper as mh

# Load the pre-trained model
calibrated_model = mh.load_model_from_db('xgboost_covid_model.pkl', 'predict_mortality_xgb')
    
def predict_mortality():
    # Load the trained model from the database
    model = mh.load_model_from_db('xgboost_covid_model.pkl', 'predict_mortality_xgb')

    # Streamlit app
    st.title('Mortality Prediction')

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
        if 'RESULT' in input_df.columns:
            input_df = input_df.drop(columns=['RESULT'])

        # Recheck the columns of the data after dropping unnecessary columns
        st.subheader('Columns after Dropping Unnecessary Columns')
        st.write(input_df.columns.tolist())

        # Make prediction
        try:
            # Make prediction
            prediction = model.predict(input_df)
            prediction_prob = model.predict_proba(input_df)

            # Display the predictions
            st.subheader('Prediction')
            input_df['Mortality Prediction'] = prediction
            st.write(input_df[['Mortality Prediction']])

            st.subheader('Prediction Probability')
            st.write(pd.DataFrame(prediction_prob, columns=[f'Probability of class {i}' for i in range(prediction_prob.shape[1])]))
        except ValueError as e:
            st.error(f"Error in prediction: {e}")
            st.write("Please ensure the uploaded file has the correct format and feature columns.")
    else:
        st.info("Please upload a CSV file to proceed.")


def mortality():
    st.subheader("Mortality")
    st.sidebar.header('Filters')

    # Load the datasets
    deaths_malaysia = mh.load_data_from_db('deaths_malaysia', 'CovidData')
    deaths_state = mh.load_data_from_db('deaths_state', 'CovidData')

    # Convert date columns to datetime
    deaths_malaysia['date'] = pd.to_datetime(deaths_malaysia['date'])
    deaths_state['date'] = pd.to_datetime(deaths_state['date'])

    selected_state = st.sidebar.selectbox('Select State', ['All'] + sorted(deaths_state['state'].unique()))
    selected_year = st.sidebar.selectbox('Select Year', ['All'] + sorted(deaths_malaysia['date'].dt.year.unique()))

    if selected_state == 'All':
        st.subheader("Overall Mortality in Malaysia")
        if selected_year == 'All':
            filtered_data = deaths_malaysia
            filtered_state_data = deaths_state
        else:
            filtered_data = deaths_malaysia[deaths_malaysia['date'].dt.year == int(selected_year)]
            filtered_state_data = deaths_state[deaths_state['date'].dt.year == int(selected_year)]
        
        # Group data by date and sum the total deaths
        grouped_data = filtered_data.groupby('date', as_index=False).sum()

        mortality_chart = alt.Chart(grouped_data).mark_line().encode(
            x='date:T',
            y='deaths_new:Q',
            tooltip=['date:T', 'deaths_new:Q']
        ).properties(
            width=700,
            height=400
        )
        st.altair_chart(mortality_chart)
        
        # Display deaths over time for all states
        st.subheader('Deaths Over Time by State')
        deaths_chart = alt.Chart(filtered_state_data).mark_line().encode(
            x='date:T',
            y='deaths_new:Q',
            color='state:N',
            tooltip=['date:T', 'state:N', 'deaths_new:Q']
        ).properties(
            width=700,
            height=400
        )
        st.altair_chart(deaths_chart)

    else:
        # Display data for the selected state
        st.subheader(f"Displaying Total Deaths for {selected_state}")
        state_data = deaths_state[deaths_state['state'] == selected_state]

        if selected_year != 'All':
            state_data = state_data[state_data['date'].dt.year == int(selected_year)]

        # New deaths in selected state
        st.subheader(f"Deaths in {selected_state}")
        state_chart = alt.Chart(state_data).mark_line().encode(
            x='date:T',
            y='deaths_new:Q',
            tooltip=['date:T', 'deaths_new:Q']
        ).properties(
            width=700,
            height=400
        )
        st.altair_chart(state_chart)

        # Display deaths due to BID (Brought-in-Dead)
        st.subheader(f"Deaths BID in {selected_state}")
        state_bid_chart = alt.Chart(state_data).mark_line().encode(
            x='date:T',
            y='deaths_bid:Q',
            tooltip=['date:T', 'deaths_bid:Q']
        ).properties(
            width=700,
            height=400
        )
        st.altair_chart(state_bid_chart)

        # Comparison of deaths_new and deaths_bid for selected state
        st.subheader(f"Comparison of New Deaths and BID Deaths for {selected_state}")

        comparison_chart = alt.Chart(state_data).transform_fold(
            ['deaths_new', 'deaths_bid'],
            as_=['Category', 'Value']
        ).mark_line().encode(
            x='date:T',
            y='Value:Q',
            color=alt.Color('Category:N', scale=alt.Scale(domain=['deaths_new', 'deaths_bid'], range=['blue', 'red']), legend=alt.Legend(title="Category")),
            tooltip=['date:T', 'Category:N', 'Value:Q']
        ).properties(
            width=700,
            height=400
        ).interactive()
        
        st.altair_chart(comparison_chart)
