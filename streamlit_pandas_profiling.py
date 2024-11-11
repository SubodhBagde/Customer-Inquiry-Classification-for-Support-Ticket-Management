import streamlit as st
import pandas as pd
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

st.header('`streamlit_pandas_profiling`')

# Load the dataset
df = pd.read_csv('https://raw.githubusercontent.com/SubodhBagde/Customer-Inquiry-Classification-for-Support-Ticket-Management/refs/heads/main/complaints_category_dataset.csv')

# Generate profile report
pr = ProfileReport(df, title="Pandas Profiling Report")

# Display the report in Streamlit
st_profile_report(pr)
