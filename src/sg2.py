import streamlit as st
import pandas as pd
import io
import google.generativeai as genai
from time import sleep

# Configure Gemini API (Replace with your actual API key)
genai.configure(api_key="AIzaSyAcUybVnEh95L57EMN5ToEMNjFsal2O3MA")


def generate_synthetic_data(user_prompt, num_rows, num_columns):
    """
    Generate synthetic data based on a user prompt using the Gemini API.

    Args:
        prompt (str): The user prompt describing the dataset.
        num_rows (int): Number of rows to generate.
        num_columns (int): Number of columns to generate.

    Returns:
        pd.DataFrame: A DataFrame containing the synthetic data, or None on error.
    """
    model = genai.GenerativeModel("models/gemini-1.0-pro")
    complete_prompt = f"""
    Generate a realistic and professional dataset based on the following description:
    {user_prompt}
    
    The dataset should contain {num_rows} rows and {num_columns} columns. Generate the data according to the number of {num_columns} columns and  {num_rows} rows. The data to be relative to {user_prompt}. Provide the data in CSV format, ensuring that all entries are professional, consistent, and realistic.
    """

    try:
        response = model.generate_content(complete_prompt)
        response_text = response.text.strip()
        # Debug: Log the raw response
        st.text_area("Raw Response from API:", response_text, height=300)

        # Attempt to parse as CSV
        csv_io = io.StringIO(response_text)
        df = pd.read_csv(csv_io)

        # Ensure the generated data has the correct number of rows and columns
        if len(df.columns) != num_columns or len(df) != num_rows:
            raise ValueError(f"Expected {num_rows} rows and {num_columns} columns, but got {df.shape}")
        return df

    except Exception as e:
        st.error(f"Error processing generated data: {e}")
        return None



def main():
    st.title("Advanced Synthetic Data Generator")

    # Input Table Name
    table_name = st.text_input("Enter Table Name:", "Synthetic Data")

    # Description of the Table
    table_description = st.text_area("Enter a description for the table:", "This table is for synthetic data generation.")

    # Number of Rows and Columns
    col1, col2 = st.columns(2)
    with col1:
        num_rows = st.number_input("Number of Rows:", min_value=1, value=10)
    with col2:
        num_columns = st.number_input("Number of Columns:", min_value=1, value=3)

    # User Prompt
    user_prompt = st.text_area(
        "Enter your prompt for generating data:",
        "Example: Generate data for list of patient's ECG report in a hospital"
    )
    
    data = None # Initialize data as None to use it later
    
    # Generate Data Button
    if st.button("Generate Data"):
        if not user_prompt.strip():
            st.error("Please enter a valid prompt.")
        else:
            with st.spinner("Generating data..."):
                sleep(1) # Show a wait spinner while generating
                data = generate_synthetic_data(user_prompt, num_rows, num_columns)

                if data is not None:
                    st.success("Data generated successfully!")

    # Preview and Download after Data Generation
    if data is not None:
        st.write("### Preview of Generated Data")
        st.dataframe(data) #Show full dataframe

        # Prepare CSV for download
        csv_buffer = io.StringIO()
        data.to_csv(csv_buffer, index=False)
        
        st.download_button(
            label="Download Data as CSV",
            data=csv_buffer.getvalue(),
            file_name=f"{table_name.replace(' ', '_').lower()}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()