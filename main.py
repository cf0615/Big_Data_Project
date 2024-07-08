# -*- coding: utf-8 -*-
import streamlit as st
from covidCase import covid_case_page
from icuPage import icuPage
#from death import deathPage

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
    
    if (selected_page == "Covid Cases"):
        covid_case_page()
    elif (selected_page == "ICU"):
        icuPage()
    else:
        death()
        
if __name__ == "__main__":
    main()
