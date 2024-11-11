import streamlit as st
import pandas as pd
import pickle

# Load the model
with open('best_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Streamlit UI
st.title("Ticket Inquiry System")

# User input for prediction
st.subheader("Predict Ticket Category")
user_input = st.text_area("Enter your inquiry here:")

if st.button("Predict Category"):
    if user_input:
        predicted_category = model.predict([user_input])[0]  # Assuming the model takes a list
        st.success(f"The predicted category for your inquiry is: **{predicted_category}**")
    else:
        st.error("Please enter an inquiry to predict the category.")

# Sample tickets data
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
    'Ticket': [
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

# Create a DataFrame
tickets_df = pd.DataFrame(tickets_data)

# Filter tickets by category
category_filter = st.selectbox("Select a category to filter tickets:", 
                                ['All'] + list(tickets_df['Category'].unique()))

if category_filter != 'All':
    filtered_tickets = tickets_df[tickets_df['Category'] == category_filter]
else:
    filtered_tickets = tickets_df

st.subheader("Sample Tickets")
st.write(filtered_tickets)

# Run the app with: streamlit run your_script.py