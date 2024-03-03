import streamlit as st
import pandas as pd
import numpy as np
import warnings
from prediction import predict
warnings.filterwarnings('ignore')

np.float = float    
np.int = int   #module 'numpy' has no attribute 'int'
np.object = object    #module 'numpy' has no attribute 'object'
np.bool = bool    #module 'numpy' has no attribute 'bool'

annsurv_df = pd.read_csv('data/Annual_survival.csv')

st.title('Survival prediction of Mangrove using supervised Machine Learning')
st.markdown('This model is to predict survival of mangrove tree based on \
            the synthetic data generated for the Kingdom of Saudi Arabia \
            factoring thier species, growth rate, log size, landscape \
            and patch type.')

st.header("Plant Features")
col1, col2 = st.columns(2)

with col1:
    species = st.selectbox("Pick your species", annsurv_df["Species"].unique())
    species = 0 if species == 'Avicennia marina' else (1 if species == 'Rhizophora mucronata' else 2)
    log_size = st.slider('Log size (cm)', 1.0, 10.0, 4.0)

with col2:
    patch_type = st.selectbox("Select patch type", annsurv_df["Patch_type"].unique())
    patch_type = 0 if patch_type == 'Connected' else (1 if patch_type == 'Rectangular' else 2)
    growth = st.slider('Growth rate (cm3)', 1.0, 13.0, 3.0)

st.text('')
if st.button("Predict survival"):
    result = predict(
        np.array([[species, log_size, patch_type, growth]]))
    resultVal = 'Prediction result : Great chance of survival' if result[0] == 1 else 'Prediction result : Unlikely to survive'
    st.text(resultVal)

st.text('')
st.text('')
st.markdown(
    '`Create by` <a href="mailto:contact.datartist@gmail.com">Datartist</a>', unsafe_allow_html=True)
#st.markdown(
#    '`Create by` <a href="mailto:contact.datartist@gmail.com">Datartist</a> | \
#         `Prediction Model used:` Random Forest Classifier', unsafe_allow_html=True)