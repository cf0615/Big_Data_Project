# -*- coding: utf-8 -*-
import streamlit as st

def covidCase():
    st.subheader("Covid-19 Cases in Malaysia")

def icu():
    st.subheader("ICU")
    
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
    #sidebar option
    selected_state = st.sidebar.selectbox('Select State', ['All'])
    
    if (selected_page == "Covid Cases"):
        covidCase()
    elif (selected_page == "ICU"):
        icu()
    else:
        death()
        
main()