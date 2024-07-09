# -*- coding: utf-8 -*-
import mongodb_helper as mh
import pandas as pd
import json

#to upload all dataset into the mongodb
cases_malaysia = pd.read_csv('PATH/TO/cases_malaysia_preprocessed.csv')
case_state = pd.read_csv('PATH/TO/caseState_preprocessed.csv')
deaths_malaysia = pd.read_csv('PATH/TO/deaths_malaysia_preprocessed.csv')
deaths_state = pd.read_csv('PATH/TO/deaths_state.csv')
icu = pd.read_csv('PATH/TO/icu_preprocessed.csv')

# Path to the GeoJSON file
geojson_path = 'PATH/TO/malaysia.geojson'  
with open(geojson_path) as f:
    geojson_data = json.load(f)
    
#Save data to mongodb
mh.save_data_to_db(cases_malaysia, 'cases_malaysia', 'CovidData')
mh.save_data_to_db(case_state, 'cases_state', 'CovidData')
mh.save_data_to_db(geojson_data, 'malaysia_geojson', 'CovidData')
mh.save_data_to_db(deaths_malaysia, 'deaths_malaysia', 'CovidData')
mh.save_data_to_db(icu, 'icu', 'CovidData')
mh.save_data_to_db(deaths_state, 'deaths_state', 'CovidData')