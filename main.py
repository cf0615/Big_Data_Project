# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import json
import joblib
from sklearn.metrics import classification_report
import altair as alt
from deathPage import mortality, predict_mortality

#load datasets
cases_malaysia = pd.read_csv('C:/Users/shenhao/OneDrive/Inti/Degree/Sem 6/Big Data/dataset/Preprocessed/cases_malaysia_preprocessed.csv')
case_state = pd.read_csv('C:/Users/shenhao/OneDrive/Inti/Degree/Sem 6/Big Data/dataset/Preprocessed/caseState_preprocessed.csv')
mortality_data = pd.read_csv('C:/Users/shenhao/OneDrive/Inti/Degree/Sem 6/Big Data/dataset/Preprocessed/covidDataPreprocessed.csv')
deaths_malaysia = pd.read_csv('C:/Users/shenhao/OneDrive/Inti/Degree/Sem 6/Big Data/dataset/mortality/deaths_malaysia.csv')
deaths_state = pd.read_csv('C:/Users/shenhao/OneDrive/Inti/Degree/Sem 6/Big Data/dataset/mortality/deaths_state.csv')

# Add tooltip data to the DataFrame
case_state['tooltip'] = case_state.apply(lambda row: f"{row['state']}: {row['cases_new']} cases", axis=1)


        
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
