import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import numpy as np
from tensorflow.keras.models import load_model

#Load the trained model
model = load_model('regression_model.h5')

with open('label_encoder_regression.pkl','rb') as f:
    label_encoder_gender = pickle.load(f)

with open('one_hot_regression.pkl','rb') as f:
    one_hot_geo = pickle.load(f)

with open('scaler.pkl','rb') as f:
    scaler = pickle.load(f)

#Streamlit app
st.title('Salary Prediction App')

#Input fields
geography = st.selectbox('Geography',one_hot_geo.categories_[0])
gender = st.selectbox('Gender',label_encoder_gender.classes_)
age = st.slider('Age',18,92)
credit_score = st.number_input('Credit Score')
tenure = st.slider('Tenure',0,10)
balance = st.number_input('Balance')
num_products = st.slider('Number of Products',1,4)
has_cr_card = st.selectbox('Has Credit Card',[0,1])
is_active_member = st.selectbox('Is Active Member',[0,1])
exited = st.selectbox('Exited',[0,1])

#Prepare the input data
input_data = pd.DataFrame({
    'CreditScore':[credit_score],
    'Geography':[geography],
    'Gender':[label_encoder_gender.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'Exited':[exited]
})

#One-hot encode the geography
geo_encoded = one_hot_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=one_hot_geo.get_feature_names_out())

input_df = pd.concat([input_data.drop('Geography',axis=1), geo_encoded_df], axis=1)

#Scale the input data
input_scaled = scaler.transform(input_df)

#Make the prediction
prediction = model.predict(input_scaled)
predicted_salary = float(prediction[0][0])

#Display the result
st.write(f'Estimated Salary: ${predicted_salary:,.2f}')
