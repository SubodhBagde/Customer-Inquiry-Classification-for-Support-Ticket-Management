import streamlit as st
import pandas as pd
import pickle
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

# Load the dataset
data_path = 'customer_inquiries.csv'
data = pd.read_csv(data_path)

# Load the model
with open('rf_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit Title
st.title("Customer Inquiry Classification for Support Ticket Management Dashboard")

tabs = st.tabs(["Classify", "Ticket Inquires/Complaints", "Insights", "Dataset Statistics"])

with tabs[0]:
    st.header("Classify Inquiry Category")
    user_input = st.text_area("Enter your inquiry:")
    if st.button("Classify", key='predict_button'):
        if user_input.strip():
            prediction = model.predict([user_input])[0]
            st.write(f"The entered inquiry lies in **{prediction}** category.")
        else:
            st.write("Please enter an inquiry to get a prediction.")

with tabs[1]:
    st.header("Ticket Inquiries Overview")
    category_counts = data['Category'].value_counts().reset_index()
    category_counts.columns = ['Category', 'Complaint']
    st.table(category_counts)

with tabs[2]:
    st.header("Inquiry Category Insights")
 
    st.subheader("Most Common Words in Inquiries")
    text_column = 'Complaint'
    text_data = data[text_column].dropna().astype(str)
    wordcloud = WordCloud(width=800, height=400).generate(" ".join(text_data))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt)

    st.subheader("Most Common Types of Inquiry Categories")
    category_counts = data['Category'].value_counts().head(10)  # Adjust based on dataset category column
    plt.figure(figsize=(10, 5))
    sns.barplot(x=category_counts.values, y=category_counts.index, palette='coolwarm')
    plt.title("Top 10 Most Common Inquiry Categories")
    plt.xlabel("Number of Inquiries")
    plt.ylabel("Category")
    st.pyplot(plt)
    
with tabs[3]:
    st.header('`streamlit_pandas_profiling`')
    df = pd.read_csv('https://raw.githubusercontent.com/SubodhBagde/Customer-Inquiry-Classification-for-Support-Ticket-Management/refs/heads/main/customer_inquiries.csv')
    pr = ProfileReport(df, title="Pandas Profiling Report")
    st_profile_report(pr)