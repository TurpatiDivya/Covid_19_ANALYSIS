import streamlit as st
import pickle
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

# Load the trained model using pickle
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Title of the web app
st.title('COVID-19 Cases Prediction')

# Sidebar for user inputs
st.sidebar.header('Input Features')

# Create input fields for the user to provide data
def user_input_features():
    date_reported = st.sidebar.date_input('Date Reported', datetime(2020, 1, 1))
    country = st.sidebar.text_input('Country', 'Afghanistan')
    continent = st.sidebar.selectbox('Continent', ('Asia', 'Africa', 'Europe', 'North America', 'South America', 'Oceania'))
    who_region = st.sidebar.selectbox('WHO Region', ('EMRO', 'AFRO', 'EURO', 'PAHO', 'SEARO', 'WPRO'))
    
    # Convert date to ordinal (to match the model training)
    date_reported = date_reported.toordinal()
    
    # Store the input data into a dictionary
    data = {
        'Date_reported': date_reported,
        'Country': country,
        'Continent': continent,
        'WHO_region': who_region
    }
    
    # Convert the dictionary into a DataFrame
    features = pd.DataFrame(data, index=[0])
    return features

# Store the user input data
input_df = user_input_features()

# Recreate the LabelEncoders for categorical features based on known data
label_encoder_country = LabelEncoder()
label_encoder_continent = LabelEncoder()
label_encoder_who_region = LabelEncoder()

# Fit the encoders with the known categories used during training
countries = ['Afghanistan', 'United States', 'India']  # Add more countries as needed
continents = ['Asia', 'Africa', 'Europe', 'North America', 'South America', 'Oceania']
who_regions = ['EMRO', 'AFRO', 'EURO', 'PAHO', 'SEARO', 'WPRO']

label_encoder_country.fit(countries)
label_encoder_continent.fit(continents)
label_encoder_who_region.fit(who_regions)

# Encoding the categorical features
input_df['Country_encoded'] = label_encoder_country.transform([input_df['Country'][0]])
input_df['Continent_encoded'] = label_encoder_continent.transform([input_df['Continent'][0]])
input_df['WHO_region_encoded'] = label_encoder_who_region.transform([input_df['WHO_region'][0]])

# Select only the relevant columns for prediction (based on how the model was trained)
X_input = input_df[['Date_reported', 'Country_encoded', 'Continent_encoded', 'WHO_region_encoded']]

# Prediction
prediction = model.predict(X_input)

# Display the prediction
st.subheader('Predicted New COVID-19 Cases:')
st.write(int(prediction[0]))
