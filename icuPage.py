# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 17:06:37 2024

@author: User
"""
import streamlit as st
import pandas as pd
import altair as alt

icu_data = pd.read_csv('C:/Users/User/.spyder-py3/icu_preprocessed.csv')

icu_data['tooltip'] = icu_data.apply(lambda row: f"{row['state']}: {row['beds_icu_total']} cases", axis=1)

def icu():
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