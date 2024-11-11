import streamlit as st
import pandas as pd
import pickle
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data_path = 'complaints_category_dataset.csv'
data = pd.read_csv(data_path)

# Load the model
with open('best_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit Title
st.title("Customer Support Inquiry Dashboard")

# Tabs for the UI sections
tabs = st.tabs(["Predict", "Ticket Inquires/Complaints", "Insights"])

# Predict Tab
with tabs[0]:
    st.header("Predict Inquiry Category")
    user_input = st.text_area("Enter your inquiry:")
    if st.button("Predict", key='predict_button'):
        if user_input.strip():
            prediction = model.predict([user_input])[0]
            st.write(f"The predicted category is: **{prediction}**")
        else:
            st.write("Please enter an inquiry to get a prediction.")

# Sample Tickets Tab
with tabs[1]:
    # Display table with Inquiry and Category columns
    tickets_data = {
    'Category': [
        'Account Issues', 'Account Issues', 'Account Issues',
        'Billing', 'Billing', 'Billing',
        'Product Issues', 'Product Issues', 'Product Issues',
        'Delivery Issues', 'Delivery Issues', 'Delivery Issues',
        'Refund', 'Refund', 'Refund',
        'Warranty', 'Warranty', 'Warranty',
        'Quality', 'Quality', 'Quality',
        'Order Issues', 'Order Issues', 'Order Issues',
        'Miscellaneous', 'Miscellaneous', 'Miscellaneous'
    ],
    ' Inquires/Complaints': [
        "I need help fixing the software issue on my device.",
        "I'm unable to log into my account, please assist.",
        "There's a bug in the system, I need it fixed.",
        "The charges on my bill don't match my usage.",
        "I was billed for an order I didn't place.",
        "I was overcharged by $100 for a service I didn't use.",
        "The product I received does not match the description.",
        "I received a product that was expired.",
        "The product arrived late and was in poor condition.",
        "The delivery company left my package outside in the rain.",
        "My package was delivered to a neighbor by mistake.",
        "The product was supposed to arrive a week ago but hasn't yet.",
        "How can I get a refund for my order?",
        "I'm requesting a refund for a product that broke after one use.",
        "I would like to initiate a refund process for my purchase.",
        "Can you send me a warranty claim form?",
        "I'm experiencing issues with a product covered by warranty, need help.",
        "I need assistance with a warranty claim for my purchase.",
        "I am not satisfied with the quality of the product I received.",
        "The item I purchased is of low quality, please help.",
        "I am disappointed by the quality of the item I received.",
        "I want to change an item in my order before it ships.",
        "The order was not delivered on time, I need assistance.",
        "The order arrived with missing parts, please assist.",
        "How can I update my personal information on your platform?",
        "Can I receive a gift card for my purchase?",
        "I need help understanding your return policy."
    ]
}
    st.dataframe(tickets_data)  # Adjust based on dataset columns

# Insights Tab
with tabs[2]:
    st.header("Insights into Common Inquiry Types")
    
    # Generate common inquiry types and associated keywords (assuming 'Inquiry' column exists)
    inquiries = data['Categories'].dropna().astype(str).tolist()  # Drop NaNs and convert to strings
    keywords = [word for inquiry in inquiries for word in inquiry.split()]
    common_keywords = Counter(keywords).most_common(10)

    # Word Cloud for visual insights
    st.subheader("Keyword Word Cloud")
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(keywords))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)

    # Most Common Inquiry Categories
    st.subheader("Most Common Types of Inquiry Categories")
    category_counts = data['Categories'].value_counts().head(10)  # Adjust based on dataset category column
    plt.figure(figsize=(10, 5))
    sns.barplot(x=category_counts.values, y=category_counts.index, palette='coolwarm')
    plt.title("Top 10 Most Common Inquiry Categories")
    plt.xlabel("Number of Inquiries")
    plt.ylabel("Category")
    st.pyplot(plt)
