# -*- coding: utf-8 -*-
# covidCase.py
import streamlit as st
import pandas as pd
import json
import folium
from streamlit_folium import folium_static
import altair as alt
import matplotlib.pyplot as plt

@st.cache_data
def load_data():
    cases_malaysia = pd.read_csv('C:/Users/junch/OneDrive/Documents/BigData/Project/Preprocess/cases_malaysia_preprocessed.csv')
    case_state = pd.read_csv('C:/Users/junch/OneDrive/Documents/BigData/Project/Preprocess/caseState_preprocessed.csv', parse_dates=['date'])
    case_state['tooltip'] = case_state.apply(lambda row: f"{row['state']}: {row['cases_new']} cases", axis=1)
    return cases_malaysia, case_state

def plot_overall_cases(cases_malaysia, selected_year):
    st.subheader("Overall Malaysia Data")
    if selected_year == 'All':
        filtered_data = cases_malaysia
    else:
        filtered_data = cases_malaysia[cases_malaysia['year'] == int(selected_year)]

    case_malaysia_chart = alt.Chart(filtered_data).mark_line().encode(
        x='date:T',
        y='cases_new:Q',
        tooltip=['date:T', 'cases_new:Q']
    ).properties(
        width=700,
        height=400
    )
    st.altair_chart(case_malaysia_chart)

def plot_cases_by_state(case_state, selected_state, selected_year):
    if selected_state != 'All':
        case_state = case_state[case_state['state'] == selected_state]
    if selected_year != 'All':
        case_state = case_state[case_state['year'] == int(selected_year)]

    st.subheader(f"COVID-19 Cases in {selected_state}")
    state_chart = alt.Chart(case_state).mark_line().encode(
        x='date:T',
        y='cases_new:Q',
        tooltip=['date:T', 'cases_new:Q']
    ).properties(
        width=700,
        height=400
    )
    st.altair_chart(state_chart)

# Add trend analysis function
def plot_case_trends(case_state):
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

def display_choropleth(case_state, selected_year):
    st.subheader("Covid-19 Cases by State")
    
    geojson_path = 'C:/Users/junch/OneDrive/Documents/BigData/Project/Dashboard/malaysia.geojson'  # Path to the GeoJSON file
    try:
        with open(geojson_path) as f:
            geojson_data = json.load(f)
    except Exception as e:
        st.error(f"Error loading GeoJSON file: {e}")
        return

    if selected_year == 'All':
        state_cases = case_state.groupby('state')['cases_new'].sum().reset_index()
    else:
        state_cases = case_state[case_state['year'] == int(selected_year)].groupby('state')['cases_new'].sum().reset_index()

    for feature in geojson_data['features']:
        state_name = feature['properties']['shapeName']
        total_cases = state_cases[state_cases['state'] == state_name]['cases_new'].values
        feature['properties']['total_cases'] = int(total_cases[0]) if len(total_cases) > 0 else 0

    m = folium.Map(location=[4.2105, 101.9758], zoom_start=6)

    try:
        folium.Choropleth(
            geo_data=geojson_data,
            data=state_cases,
            columns=['state', 'cases_new'],
            key_on='feature.properties.shapeName',
            fill_color='YlOrRd',
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name='Number of Cases'
        ).add_to(m)

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

def covid_case_page():
    cases_malaysia, case_state = load_data()
    st.sidebar.header('Filters')
    selected_state = st.sidebar.selectbox('Select State', ['All'] + sorted(case_state['state'].unique()))
    selected_year = st.sidebar.selectbox('Select Year', ['All'] + sorted(case_state['year'].unique()))

    plot_overall_cases(cases_malaysia, selected_year)
    if selected_state == 'All':
        display_choropleth(case_state, selected_year)
        plot_case_trends(case_state)
    else:
        plot_cases_by_state(case_state, selected_state, selected_year)

        state_data = case_state[case_state['state'] == selected_state]
        if selected_year != 'All':
            state_data = state_data[state_data['year'] == int(selected_year)]

        st.subheader(f"New Cases for {selected_state}")
        case_state_chart = alt.Chart(state_data).mark_line().encode(
            x='date:T',
            y='cases_new:Q',
            tooltip=['date:T', 'cases_new:Q']
        ).properties(
            width=700,
            height=400
        )
        st.altair_chart(case_state_chart)

        st.subheader(f"Cases Recovered for {selected_state}")
        recovered_state_chart = alt.Chart(state_data).mark_line().encode(
            x='date:T',
            y='cases_recovered:Q',
            tooltip=['date:T', 'cases_recovered:Q']
        ).properties(
            width=700,
            height=400
        )
        st.altair_chart(recovered_state_chart)

        st.subheader(f"Cases Active for {selected_state}")
        active_state_chart = alt.Chart(state_data).mark_line().encode(
            x='date:T',
            y='cases_active:Q',
            tooltip=['date:T', 'cases_active:Q']
        ).properties(
            width=700,
            height=400
        )
        st.altair_chart(active_state_chart)
