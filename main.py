# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import json
import joblib
from sklearn.metrics import classification_report
import altair as alt

#load datasets
cases_malaysia = pd.read_csv('C:/Users/shenhao/OneDrive/Inti/Degree/Sem 6/Big Data/dataset/Preprocessed/cases_malaysia_preprocessed.csv')
case_state = pd.read_csv('C:/Users/shenhao/OneDrive/Inti/Degree/Sem 6/Big Data/dataset/Preprocessed/caseState_preprocessed.csv')
mortality_data = pd.read_csv('C:/Users/shenhao/OneDrive/Inti/Degree/Sem 6/Big Data/dataset/Preprocessed/covidDataPreprocessed.csv')
deaths_malaysia = pd.read_csv('C:/Users/shenhao/OneDrive/Inti/Degree/Sem 6/Big Data/dataset/mortality/deaths_malaysia.csv')
deaths_state = pd.read_csv('C:/Users/shenhao/OneDrive/Inti/Degree/Sem 6/Big Data/dataset/mortality/deaths_state.csv')

# Add tooltip data to the DataFrame
case_state['tooltip'] = case_state.apply(lambda row: f"{row['state']}: {row['cases_new']} cases", axis=1)

# Load the pre-trained model
model_path = 'C:/Users/shenhao/OneDrive/Inti/Degree/Sem 6/Big Data/xgboost_covid_model_important.pkl'  # Update with the actual path
calibrated_model = joblib.load(model_path)

# Define the threshold for prediction
threshold = 0.3

# Function to make predictions with a custom threshold
def predict_with_threshold(model, X, threshold):
    proba = model.predict_proba(X)[:, 1]
    return (proba >= threshold).astype(int), proba

def display_classification_report(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)
    
def predict_mortality():
    st.subheader("Mortality Prediction")

    # Create input fields for each feature
    pneumonia = st.selectbox('Pneumonia', [0, 1])
    age = st.number_input('Age', min_value=0, max_value=120, value=55)
    diabetes = st.selectbox('Diabetes', [0, 1])
    icu = st.selectbox('ICU', [0, 1])
    renal_chronic = st.selectbox('Renal Chronic', [0, 1])
    sex = st.selectbox('Sex (0: Male, 1: Female)', [0, 1])
    hypertension = st.selectbox('Hypertension', [0, 1])
    obesity = st.selectbox('Obesity', [0, 1])
    other_disease = st.selectbox('Other Disease', [0, 1])
    inmsupr = st.selectbox('Immunosuppressed', [0, 1])

    if st.button('Predict'):
        # Create a dataframe for the input features
        input_features = pd.DataFrame({
            'PNEUMONIA': [pneumonia],
            'AGE': [age],
            'DIABETES': [diabetes],
            'ICU': [icu],
            'RENAL_CHRONIC': [renal_chronic],
            'SEX': [sex],
            'HIPERTENSION': [hypertension],
            'OBESITY': [obesity],
            'OTHER_DISEASE': [other_disease],
            'INMSUPR': [inmsupr]
        })

        # Make prediction
        prediction, prediction_proba = predict_with_threshold(calibrated_model, input_features, threshold)
        result = 'Death' if prediction[0] == 1 else 'Survival'
        st.write(f"Prediction: {result}")
        #st.write(f"Prediction Probability: {prediction_proba[0] * 100:.2f}% chance of death")


def mortality():
    st.subheader("Mortality")
    st.sidebar.header('Filters')

    # Load the datasets
    deaths_malaysia = pd.read_csv('C:/Users/shenhao/OneDrive/Inti/Degree/Sem 6/Big Data/dataset/mortality/deaths_malaysia.csv')
    deaths_state = pd.read_csv('C:/Users/shenhao/OneDrive/Inti/Degree/Sem 6/Big Data/dataset/mortality/deaths_state.csv')

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
        
def covidCase():
    st.subheader("Covid-19 Cases in Malaysia")
    
    # Sidebar option
    selected_state = st.sidebar.selectbox('Select State', ['All'] + sorted(case_state['state'].unique()))
    selected_year = st.sidebar.selectbox('Select Year', ['All'] + sorted(case_state['year'].unique())) 
    
    # Check if the dataframes are loaded
    if 'cases_malaysia' not in globals() or 'case_state' not in globals():
        st.error("Dataframes are not loaded. Please load the data.")
        return
    
    if selected_state == 'All':
        # Display overall Malaysia data
        st.write("Overall Malaysia Data")
        st.line_chart(cases_malaysia[['date', 'cases_new']].set_index('date'))
        
        # Display choropleth map
        st.write("Covid-19 Cases by State")
        
        # Load GeoJSON data
        geojson_path = 'C:/Users/shenhao/Downloads/malaysia.geojson'  # Path to the uploaded GeoJSON file
        try:
            with open(geojson_path) as f:
                geojson_data = json.load(f)
        except Exception as e:
            st.error(f"Error loading GeoJSON file: {e}")
            return
        
        # Create a folium map
        m = folium.Map(location=[4.2105, 101.9758], zoom_start=6)
        
        # Create choropleth
        try:
            folium.Choropleth(
                geo_data=geojson_data,
                data=case_state,
                columns=['state', 'cases_new'],
                key_on='feature.properties.shapeName',  # Ensure this matches the GeoJSON property for state names
                fill_color='YlOrRd',
                fill_opacity=0.7,
                line_opacity=0.2,
                legend_name='Number of Cases'
            ).add_to(m)
            
            # Add tooltip for cases
            folium.GeoJson(
                geojson_data,
                style_function=lambda feature: {
                    'fillColor': 'transparent',
                    'color': 'transparent'
                },
                tooltip=folium.GeoJsonTooltip(
                    fields=['shapeName'],
                    aliases=['State: '],
                    localize=True,
                    sticky=False,
                    labels=True,
                    style="""
                        background-color: #F0EFEF;
                        border: 2px solid black;
                        border-radius: 3px;
                        box-shadow: 3px;
                    """
                )
            ).add_to(m)
            
            folium_static(m)
        except Exception as e:
            st.error(f"Error creating choropleth map: {e}")
    else:
        # Display data for the selected state
        st.write(f"Displaying Covid-19 cases for {selected_state}")
        state_data = case_state[case_state['state'] == selected_state]
        st.line_chart(state_data[['date', 'cases_new']].set_index('date'))


def icu():
    st.subheader("ICU")
    #sidebar option
    selected_state = st.sidebar.selectbox('Select Year', ['All'])
    
def death():
    st.subheader("Death")

def main():
    #page title
    st.title("Covid-19")
    #sidebar title
    st.sidebar.header('Pages')
    #sidebar option
    selected_page = st.sidebar.selectbox('Select Page', ['Covid Cases', 'ICU', 'Deaths', 'Mortality', 'Mortality Prediction'])
    #markdown
    st.sidebar.markdown("---")
    #sidebar filter
    st.sidebar.header('Filter')
    
    
    if (selected_page == "Covid Cases"):
        covidCase()
    elif (selected_page == "ICU"):
        icu()
    elif selected_page == "Deaths":
        death()
    elif selected_page == "Mortality":
        mortality()
    elif selected_page == "Mortality Prediction":
        predict_mortality()
        
if __name__ == "__main__":
    main()
