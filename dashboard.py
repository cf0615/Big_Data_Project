import streamlit as st
import pandas as pd
import altair as alt

# Load the datasets
cases_df = pd.read_csv("C:/Users/junch/OneDrive/Documents/BigData/Project/Preprocess/cases_malaysia_preprocessed.csv")
icu_df = pd.read_csv("C:/Users/junch/OneDrive/Documents/BigData/Project/Preprocess/icu_preprocessed.csv")

# Convert date columns to datetime
cases_df['date'] = pd.to_datetime(cases_df['date'])
icu_df['date'] = pd.to_datetime(icu_df['date'])

# Dashboard title
st.title('COVID-19 Dashboard for Malaysia')

# Sidebar for user input
st.sidebar.header('Filter Options')

# Filter options
selected_state = st.sidebar.selectbox('Select State', ['All'] + sorted(icu_df['state'].unique()))
selected_year = st.sidebar.selectbox('Select Year', ['All'] + sorted(cases_df['year'].unique()))

# Filter the data based on user input
if selected_state != 'All':
    icu_df = icu_df[icu_df['state'] == selected_state]

if selected_year != 'All':
    cases_df = cases_df[cases_df['year'] == int(selected_year)]
    icu_df = icu_df[icu_df['year'] == int(selected_year)]

# Display new cases over time
st.subheader('New COVID-19 Cases Over Time')
cases_chart = alt.Chart(cases_df).mark_line().encode(
    x='date:T',
    y='cases_new:Q',
    tooltip=['date:T', 'cases_new:Q']
).properties(
    width=700,
    height=400
)
st.altair_chart(cases_chart)

# Display ICU usage over time
st.subheader('ICU Usage Over Time')
icu_chart = alt.Chart(icu_df).mark_line().encode(
    x='date:T',
    y='icu_covid:Q',
    color='state:N',
    tooltip=['date:T', 'state:N', 'icu_covid:Q']
).properties(
    width=700,
    height=400
)
st.altair_chart(icu_chart)

# Display ICU bed availability
st.subheader('ICU Bed Availability')
icu_beds_chart = alt.Chart(icu_df).mark_bar().encode(
    x='state:N',
    y='beds_icu_total:Q',
    color='state:N',
    tooltip=['state:N', 'beds_icu_total:Q']
).properties(
    width=700,
    height=400
)
st.altair_chart(icu_beds_chart)

# Display a summary table of the selected data
st.subheader('Summary of Selected Data')
st.write(cases_df if selected_state == 'All' else icu_df)
