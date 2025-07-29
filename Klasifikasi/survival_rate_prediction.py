# import library
import streamlit as st
import numpy as np
import pandas as pd
import pickle

#Judul Utama
st.title('Survival Rate Predictor')
st.text('This web can be used to predict your survival rate')



# Menambahkan sidebar
st.sidebar.header("Please input your features")

def create_user_input():
    
    # Numerical Features
    pclass = st.sidebar.slider('pclass', min_value=1, max_value=3, value=1)
    age = st.sidebar.slider('age', min_value=0, max_value=80, value=25)
    sibsp = st.sidebar.slider('sibsp', min_value=0, max_value=8, value=0)
    parch = st.sidebar.slider('parch', min_value=0, max_value=8, value=0)
    fare = st.sidebar.number_input('fare', min_value=0, max_value=513, value=100)
    

    # Categorical Features
    sex = st.sidebar.radio('sex', ['female', 'male'])
    embarked = st.sidebar.radio('embarked', ['Q', 'C','S'])
    deck=st.sidebar.radio('deck', ['C', 'E', 'G', 'D', 'A', 'B', 'F'])
    

    # Creating a dictionary with user input
    user_data = {
        'pclass': pclass,
        'age': age,
        'sibsp': sibsp,
        'parch': parch,
        'age': age,
        'fare': fare,
        'sex':sex,
        'embarked':embarked,
        'deck':deck
    }
    
    # Convert the dictionary into a pandas DataFrame (for a single row)
    user_data_df = pd.DataFrame([user_data])
    
    return user_data_df

# Get customer data
data_customer = create_user_input()

# Membuat 2 kontainer
col1, col2 = st.columns(2)

# Kiri
with col1:
    st.subheader("Customer's Features")
    st.write(data_customer.transpose())

# Load model
with open(r'/mount/src/deploy_model/Klasifikasi/final_model_complete.pkl', 'rb') as f:
    model_loaded = pickle.load(f)
    
# Predict to data
kelas = model_loaded.predict(data_customer)
probability = model_loaded.predict_proba(data_customer)[0]  # Get the probabilities

# Menampilkan hasil prediksi

# Bagian kanan (col2)
with col2:
    st.subheader('Prediction Result')
    if kelas == 1:
        st.write('Class 1: This customer will Survive')
    else:
        st.write('Class 2: This customer will Survive')
    
    # Displaying the probability of the customer buying
    st.write(f"Probability of Survive: {probability[1]:.2f}")  # Probability of class 1 (BUY)
