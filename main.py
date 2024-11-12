import streamlit as st
import streamlit_shadcn_ui as ui
import pandas as pd
import pickle
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data_path = 'customer_inquiries.csv'
data = pd.read_csv(data_path)

# Load the model
with open('new_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit Title
st.title("Customer Inquiry Classification for Support Ticket Management Dashboard")

tabs = st.tabs(["Predict", "Ticket Inquires/Complaints", "Insights"])

with tabs[0]:
    st.header("Predict Inquiry Category")
    user_input = st.text_area("Enter your inquiry:")
    if st.button("Predict", key='predict_button'):
        if user_input.strip():
            prediction = model.predict([user_input])[0]
            st.write(f"The predicted category is: **{prediction}**")
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