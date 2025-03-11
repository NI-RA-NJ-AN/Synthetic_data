import streamlit as st
import pandas as pd
from  ctgan import CTGAN 

def generate_synthetic_data(data, discrete_columns,num_samples):
    ctgan =CTGAN(epochs=20)
    ctgan.fit(data,discrete_columns)
    synthetic_data =ctgan.sample(num_samples)
    return synthetic_data

st.title("Synthetic Data Generator")
data_file= st.file_uploader("Upload your data file in (CSV)", type="csv")

if data_file is not None:
    data = pd.read_csv(data_file)
    discrete_option = data.select_dtypes(include=['object']).columns.tolist()
    selected_discrete = st.multiselect("select discrete colmns:",options=discrete_option,default=discrete_option)
    num_samples=st.number_input("Enter the number of synthetic data points:",min_value=1)

    if st.button("Generate Synthetic Data"):
        synthetic_data= generate_synthetic_data(data,selected_discrete,num_samples)
        st.dataframe(synthetic_data)

        st.download_button(label= "Download Synthetic Data",data= synthetic_data.to_csv(), file_name= "new_Synthetic_data.csv")
st.info("**Note:** This is not Real world data, this is synthetic generated data. Please use with caution")