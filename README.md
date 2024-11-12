# Customer Inquiry Classification for Support Ticket Management

This project automates the classification of customer inquiries into predefined categories to improve support ticket management. It uses a machine learning pipeline to classify customer complaints.

[You can visit the live app here!](https://customer-inquiry-classification-for-support-ticket-management.streamlit.app/)

## Features

- Text preprocessing (removal of special characters, tokenization, stopword removal, lemmatization)
- Word and character n-grams for feature extraction
- Logistic Regression model for classification
- Streamlit UI for real-time predictions
- Deployed using Streamlit for easy access

## Dataset

The custom dataset contains two columns:
- **Complaint**: The customer inquiry text
- **Category**: The category label for the complaint

## Methodology

1. Initialize stopwords and lemmatizer.
2. Preprocess the 'Complaint' column.
3. Split the data into features (X) and target (y).
4. Use word and character n-grams.
5. Build a pipeline with TF-IDF and Logistic Regression.
6. Train the model.
7. Predict on the test set and evaluate performance.
8. Create a Streamlit UI and deploy the model.

## How to Run

1. Clone the repository:
   
   ```bash
   git clone https://github.com/your-username/customer-inquiry-classification.git
   ```
2. Install required packages:

   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```
