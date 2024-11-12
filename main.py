import streamlit as st
import streamlit_shadcn_ui as ui
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
st.title("Customer Inquiry Classification for Support Ticket Management Dashboard")

tabs = st.tabs(["Predict", "Ticket Inquires/Complaints", "Insights"])

with tabs[0]:
    st.header("Predict Inquiry Category")
    user_input = st.text_area("Enter your inquiry:")
    
    # Button for prediction
    if ui.button(text="Predict", key="predict", className="bg-orange-500 text-white"):
        if user_input.strip():
            prediction = model.predict([user_input])[0]
            # Display prediction in a card
            ui.card(
                title="Predicted Category",
                content=f"The predicted category is: **{prediction}**",
                className="bg-blue-100 text-blue-900 mt-3 p-4 shadow-md rounded-lg"
            )
        else:
            st.write("Please enter an inquiry to get a prediction.")

with tabs[1]:
    st.header("Ticket Inquiries Overview")
    category_counts = data['Categories'].value_counts().reset_index()
    category_counts.columns = ['Categories', 'Complaints']
    st.table(category_counts)

with tabs[2]:
    st.header("Inquiry Category Insights")
 
    st.subheader("Most Common Words in Inquiries")
    text_column = 'Complaints'
    text_data = data[text_column].dropna().astype(str)
    wordcloud = WordCloud(width=800, height=400).generate(" ".join(text_data))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt)

    st.subheader("Top Inquiry Categories")
    top_categories = data['Categories'].value_counts().nlargest(10)
    fig, ax = plt.subplots(figsize=(10, 6)) 
    sns.barplot(x=top_categories.index, y=top_categories.values, palette="viridis", ax=ax)

    ax.set_title("Top 10 Inquiry Categories")
    ax.set_xlabel("Category")
    ax.set_ylabel("Count")

    st.pyplot(fig) 
    plt.close(fig)