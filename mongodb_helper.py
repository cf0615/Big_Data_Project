# -*- coding: utf-8 -*-
import pymongo
import pickle
import pandas as pd

client = pymongo.MongoClient('mongodb://localhost:27017/')
db = client['Project_db']

def save_model_to_db(model, dbcollection, model_name):
    #pickling the model
    pickled_model = pickle.dumps(model)
    
    #save the model to mongodb
    #create collection
    collection = db[dbcollection]
    data_info = collection.insert_one({model_name: pickled_model, 'name': model_name})
    print(data_info.inserted_id, ' saved with this id.')

def load_model_from_db(model_name, dbcollection): 
    json_data = {}
    
    #create collection
    collection = db[dbcollection]
    data = collection.find({'name': model_name})
    
    for i in data:
        json_data = i
    
    pickled_model = json_data[model_name]
    
    return pickle.loads(pickled_model)

def save_data_to_db(data, name, dbcollection):
    #create collection
    collection = db[dbcollection]
    
    #convert dataframe to dictionary and insert into mongodb
    if isinstance(data, pd.DataFrame):
        data_dict = {
            'name': name,
            'data': data.to_dict('records')
        }
    else:
        data_dict = {
            'name': name,
            'data': data
        }
        
    collection.insert_one(data_dict)
        
def load_data_from_db(name, dbcollection):
    collection = db[dbcollection]
    
    #load data from mongodb
    result = collection.find_one({'name': name})
    
    if result and 'data' in result:
        data = result['data']
        if isinstance(data, list) and all(isinstance(item, dict) for item in data):
            # Convert list of dictionaries back to DataFrame if applicable
            return pd.DataFrame(data)
        return data
    return None