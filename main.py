# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import json
import folium
from streamlit_folium import folium_static
import altair as alt
import matplotlib.pyplot as plt
from datetime import date
import joblib
from sklearn.preprocessing import StandardScaler
import streamlit as st
import pandas as pd
import numpy as np

# Load the datasets
cases_malaysia = pd.read_csv('C:/Users/User/.spyder-py3/cases_malaysia_preprocessed.csv')
case_state = pd.read_csv('C:/Users/User/.spyder-py3/caseState_preprocessed.csv')
icu = pd.read_csv('C:/Users/User/.spyder-py3/Kaggle_Sirio_preprocessed.csv')

print(icu.columns)

# Add tooltip data to the DataFrame
case_state['tooltip'] = case_state.apply(lambda row: f"{row['state']}: {row['cases_new']} cases", axis=1)

# Add trend analysis function
def plot_case_trends():
    states = list(case_state.state.unique())
    
    plt.rcParams["font.family"] = "monospace"
    plt.rcParams.update({'font.size': 16})
    figure, axes = plt.subplots(4, 4, figsize=(30, 30), sharey=True)
    figure.set_size_inches([15, 15], forward=True)
    figure.suptitle('COVID-19 Case Trends')
    axe = axes.ravel()
    
    i = 0
    for s in states:
        temp = case_state[case_state.state == s].copy()
        temp['cases_new'] = (temp['cases_new'] - temp['cases_new'].min()) / (temp['cases_new'].max() - temp['cases_new'].min()) * 100
        temp['cases_new_ma'] = temp['cases_new'].rolling(window=7).mean()
        temp = temp.dropna()
    
        temp['cases_new'].plot(ax=axe[i], legend=None, color='black', linewidth=0.5, alpha=0.5)
        temp['cases_new_ma'].plot(ax=axe[i], legend=None, color='black')
    
        axe[i].set_title(s)
        i += 1
    
    plt.setp(axes, xticks=[], yticks=[])
    figure.tight_layout()
    figure.subplots_adjust(top=0.91)
    
    st.pyplot(figure)

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
        st.subheader("Overall Malaysia Data")
        if selected_year == 'All':
            filtered_data = cases_malaysia
        else:
            filtered_data = cases_malaysia[cases_malaysia['year'] == int(selected_year)]
        
        case_malaysia_chart = alt.Chart(filtered_data).mark_line().encode(
                x = 'date:T',
                y = 'cases_new:Q',
                tooltip=['date:T', 'cases_new:Q']
            ).properties(
                    width=700,
                    height=400
                )
        st.altair_chart(case_malaysia_chart)
        
        # Display choropleth map
        st.subheader("Covid-19 Cases by State")
        
        # Load GeoJSON data
        geojson_path = 'C:/Users/User/.spyder-py3/malaysia.geojson'  # Path to the uploaded GeoJSON file
        try:
            with open(geojson_path) as f:
                geojson_data = json.load(f)
        except Exception as e:
            st.error(f"Error loading GeoJSON file: {e}")
            return
        
        # Aggregate cases by state for the selected year
        if selected_year == 'All':
            state_cases = case_state.groupby('state')['cases_new'].sum().reset_index()
        else:
            state_cases = case_state[case_state['year'] == int(selected_year)].groupby('state')['cases_new'].sum().reset_index()
        
        # Merge the case numbers into the GeoJSON data
        for feature in geojson_data['features']:
            state_name = feature['properties']['shapeName']
            total_cases = state_cases[state_cases['state'] == state_name]['cases_new'].values
            feature['properties']['total_cases'] = int(total_cases[0]) if len(total_cases) > 0 else 0
        
        # Create a folium map
        m = folium.Map(location=[4.2105, 101.9758], zoom_start=6)
        
        # Create choropleth
        try:
            folium.Choropleth(
                geo_data=geojson_data,
                data=state_cases,
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
                    fields=['shapeName', 'total_cases'],
                    aliases=['State: ', 'Cases: '],
                    localize=True,
                    sticky=False,
                    labels=True,
                    style="""
                        background-color: #F0EFEF;
                        border: 2px solid black;
                        border-radius: 3px;
                        box-shadow: 3px;
                    """,
                    toLocaleString=True
                )
            ).add_to(m)
            
            folium_static(m)
        except Exception as e:
            st.error(f"Error creating choropleth map: {e}")
            
        # Display trend analysis
        st.subheader("COVID-19 Case Trends by State")
        plot_case_trends()
            
    else:
        # Display data for the selected state
        st.subheader(f"Displaying Covid-19 cases for {selected_state}")
        state_data = case_state[case_state['state'] == selected_state]
        
        if selected_year != 'All':
            state_data = state_data[state_data['year'] == int(selected_year)]
        
        #new case of selected state
        st.subheader(f"New Cases for {selected_state}")
        case_state_chart = alt.Chart(state_data).mark_line().encode(
                x = 'date:T',
                y = 'cases_new:Q',
                tooltip=['date:T', 'cases_new:Q']
            ).properties(
                    width=700,
                    height=400
                )
        st.altair_chart(case_state_chart)
        
        #recovered of selected state
        st.subheader(f"Cases Recovered for {selected_state}")
        recovered_state_chart = alt.Chart(state_data).mark_line().encode(
                x = 'date:T',
                y = 'cases_recovered:Q',
                tooltip=['date:T', 'cases_recovered:Q']
            ).properties(
                    width=700,
                    height=400
                )
        st.altair_chart(recovered_state_chart)
        
        #case active of selected state
        st.subheader(f"Cases Active for {selected_state}")
        active_state_chart = alt.Chart(state_data).mark_line().encode(
                x = 'date:T',
                y = 'cases_active:Q',
                tooltip=['date:T', 'cases_active:Q']
            ).properties(
                    width=700,
                    height=400
                )
        st.altair_chart(active_state_chart)

def icu():
    st.subheader("ICU")
    
    # Load the model
    model_path = 'C:/Users/User/.spyder-py3/xgb_icu_prediction.pkl'
    scaler_path = 'C:/Users/User/.spyder-py3/scaler.pkl'
    
    try:
        model = joblib.load(model_path)
        st.write(f"Model type: {type(model)}")  # Debugging: Check model type
    except FileNotFoundError:
        st.error(f"Model file not found. Please ensure '{model_path}' exists at the specified path.")
        return
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return
    
    try:
        scaler = joblib.load(scaler_path)
        st.write("Scaler loaded successfully.")  # Debugging: Confirm scaler load
    except FileNotFoundError:
        st.error(f"Scaler file not found. Please ensure '{scaler_path}' exists at the specified path.")
        return
    except Exception as e:
        st.error(f"An error occurred while loading the scaler: {e}")
        return
    
    # Define all columns based on the dataset
    dataset_path = 'C:/Users/User/.spyder-py3/Kaggle_Sirio_preprocessed.csv'
    data = pd.read_csv(dataset_path)
    all_columns = data.columns.tolist()[:-1]  # Get all column names except the last one (target variable)
    
    # Define selected columns for user inputs
    selected_columns = [
        'AGE_ABOVE65', 'GENDER', 'DISEASE GROUPING 1', 'DISEASE GROUPING 2',
        'BLOODPRESSURE_SISTOLIC_MEAN', 'BLOODPRESSURE_DIASTOLIC_MEAN', 
        'HEART_RATE_MEAN', 'RESPIRATORY_RATE_MEAN', 'TEMPERATURE_MEAN', 
        'OXYGEN_SATURATION_MEAN'
    ]
    
    # Create user input fields in the sidebar
    st.sidebar.header('User Input Features')
    user_inputs = {}
    user_inputs['AGE_ABOVE65'] = st.sidebar.selectbox('Age Above 65', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    user_inputs['GENDER'] = st.sidebar.selectbox('Gender', options=[0, 1], format_func=lambda x: 'Female' if x == 1 else 'Male')
    
    for feature in selected_columns[2:]:
        min_val = data[feature].min()
        max_val = data[feature].max()
        step_val = (max_val - min_val) / 100  # Just a simple way to define step size
        
        user_inputs[feature] = st.sidebar.number_input(f'{feature} value', min_value=float(min_val), max_value=float(max_val), step=float(step_val))
    
    # Prepare input data with all features
    input_data = {feature: 0 for feature in all_columns}
    for feature in selected_columns:
        input_data[feature] = user_inputs[feature]

    input_df = pd.DataFrame([input_data])
    
    # Ensure all column names are strings
    input_df.columns = input_df.columns.astype(str)
    
    # Standardize the input data using the loaded scaler
    standardized_input_df = scaler.transform(input_df)
    
    # Print standardized input data for debugging
    st.write("Standardized Input Data:", standardized_input_df)
    
    # Predict ICU admission using predict_proba to get probabilities
    try:
        probabilities = model.predict_proba(standardized_input_df)
        st.write("Prediction Probabilities:", probabilities)  # Debugging: Show prediction probabilities
        prediction = np.argmax(probabilities, axis=1)
        st.write("Raw Prediction Result:", prediction)  # Debugging: Show raw prediction result
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        return
    
    # Display the prediction result
    st.write(f"The predicted ICU admission outcome is: {'Yes' if prediction[0] == 1 else 'No'}")

    # Debugging: Check the model type and prediction method
    if hasattr(model, 'predict_proba'):
        st.write("The model supports predict_proba method.")
    else:
        st.write("The model does not support predict_proba method.")


def death():
    st.subheader("Death")

def main():
    #page title
    st.title("Covid-19")
    #sidebar title
    st.sidebar.header('Pages')
    #sidebar option
    selected_page = st.sidebar.selectbox('Select Page', ['Covid Cases', 'ICU', 'Deaths'])
    #markdown
    st.sidebar.markdown("---")
    #sidebar filter
    st.sidebar.header('Filter')  
    
    if (selected_page == "Covid Cases"):
        covidCase()
    elif (selected_page == "ICU"):
        icu()
    else:
        death()
        
if __name__ == "__main__":
    main()
