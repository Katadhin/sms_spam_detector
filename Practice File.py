!pip install scikit-learn
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import gradio as gr

# Define the SMS classification function
def sms_classification(df):
    # Set features and target variables
    features = df['text_message']  # Update column name here
    target = df['label']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.33, random_state=42)
    
    # Build pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LinearSVC())
    ])
    
    # Fit model to transformed training data
    model = pipeline.fit(X_train, y_train)
    
    return model

# Load the dataset into a DataFrame
sms_text_df = pd.read_csv('Resources/SMSSpamCollection.csv')

# Call the sms_classification function
model = sms_classification(sms_text_df)

# Save the model
joblib.dump(model, 'sms_spam_model.joblib')

# Load the model
model = joblib.load('sms_spam_model.joblib')

# Define the function for predicting SMS classification
def sms_prediction(text):
    # Make prediction
    prediction = model.predict([text])[0]
    if prediction == 'ham':
        return f"The text message: '{text}', is not spam."
    else:
        return f"The text message: '{text}', is spam."

# Create Gradio interface
gr.Interface(fn=sms_prediction, inputs="text", outputs="text", title="SMS Spam Detection", description="Enter a text message to determine if it's spam or not.").launch()
