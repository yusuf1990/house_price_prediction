import numpy as np
import streamlit as st
import pickle
def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data=load_model()
regressor_loaded=data['model']
latitude=data['latitude']
longitude=data['longitude']
population=data['population']

def show_predict_page_py():
    st.title('Estimating House Value')

    st.write("### We need some info to predict house price")

    t="""
    population={1000,1500,10,200}
    #population=st.number_input(value=population)#'Enter population',value=population)

    population=st.selectbox("Population",population)"""

    user_input = {
        'latitude': st.number_input('Enter Latitude', value=latitude.iloc[0]),
        'longitude': st.number_input('Enter Longitude', value=longitude.iloc[0]),
        'population': st.selectbox('Select Population', list(population.keys()))  # Use the correct variable name
    }

    # Perform prediction using the loaded model and user input
    # You can access population data using population_data[user_input['population']]
    predicted_salary = regressor_loaded.predict(np.array([user_input['latitude'], user_input['longitude'], population[user_input['population']]]).reshape(1, -1))[0]

    st.write(f'Predicted Salary: ${predicted_salary:.2f}')

