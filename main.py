# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import streamlit as st
from covidCase import covid_case_page
#from icu import icuPage
#from death import deathPage

def icu():
    st.subheader("ICU")
    #sidebar option
    #selected_state = st.sidebar.selectbox('Select Year', ['All'])
    
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
        covid_case_page()
    elif (selected_page == "ICU"):
        icu()
    else:
        death()
        
if __name__ == "__main__":
    main()
