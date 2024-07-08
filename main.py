# -*- coding: utf-8 -*-
import streamlit as st
from deathPage import mortality, predict_mortality
from covidCase import covid_case_page
from icuPage import icuPage

def main():
    #page title
    st.title("Covid-19")
    #sidebar title
    st.sidebar.header('Pages')
    #sidebar option
    selected_page = st.sidebar.selectbox('Select Page', ['Covid Cases', 'ICU', 'Deaths', 'Mortality', 'Mortality Prediction'])
    #markdown
    st.sidebar.markdown("---")
    
    if (selected_page == "Covid Cases"):
        covid_case_page()
    elif (selected_page == "ICU"):
        icuPage()
    elif selected_page == "Mortality":
        mortality()
    elif selected_page == "Mortality Prediction":
        predict_mortality()
        
if __name__ == "__main__":
    main()
