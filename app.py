import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
import pickle

# Set page configuration
st.set_page_config(
    page_title="Mental Health Status Classifier",
    page_icon="🧠",
    layout="centered"
)

@st.cache_resource
def download_nltk_resources():
    """Download required NLTK resources."""
    nltk.download('stopwords')
    return set(stopwords.words('english'))

@st.cache_resource
def load_model_and_resources():
    """Load the model, tokenizer, and label encoder."""
    try:
        # Load the fine-tuned model and tokenizer
        model = AutoModelForSequenceClassification.from_pretrained("saved_mental_status_bert")
        tokenizer = AutoTokenizer.from_pretrained("saved_mental_status_bert")
        
        # Load the label encoder
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
            
        return model, tokenizer, label_encoder
    except Exception as e:
        st.error(f"Error loading model resources: {str(e)}")
        return None, None, None

def clean_statement(statement, stop_words):
    """Clean and preprocess the input text."""
    # Convert to lowercase
    statement = statement.lower()
    
    # Remove special characters
    statement = re.sub(r'[^\w\s]', '', statement)
    
    # Remove numbers
    statement = re.sub(r'\d+', '', statement)
    
    # Tokenize and remove stopwords
    words = statement.split()
    words = [word for word in words if word not in stop_words]
    
    # Rejoin words
    return ' '.join(words)

def predict_mental_health(text, model, tokenizer, label_encoder, stop_words):
    """Predict mental health status from input text."""
    # Clean the text
    cleaned_text = clean_statement(text, stop_words)
    
    # Tokenize
    inputs = tokenizer(
        cleaned_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    
    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        predicted_class = torch.argmax(logits, dim=1).item()
    
    return predicted_class, probabilities[0].tolist(), label_encoder.inverse_transform([predicted_class])[0]

def main():
    st.title("🧠 Mental Health Status Classifier")
    st.markdown("""
    This application uses a fine-tuned BERT model to analyze text and identify potential mental health status.
    
    **Note: This is a demonstration tool and should not be used as a substitute for professional medical advice.**
    """)
    
    # Load resources
    stop_words = download_nltk_resources()
    model, tokenizer, label_encoder = load_model_and_resources()
    
    if model is None or tokenizer is None or label_encoder is None:
        st.error("Failed to load required resources. Please check the model files and try again.")
        return
    
    # Set model to evaluation mode
    model.eval()
    
    # Text input
    text_input = st.text_area(
        "Enter text to analyze:",
        height=150,
        placeholder="Type or paste text here..."
    )
    
    if st.button("Analyze"):
        if not text_input:
            st.warning("Please enter some text to analyze.")
            return
            
        with st.spinner("Analyzing..."):
            try:
                # Get prediction
                predicted_idx, probabilities, predicted_class = predict_mental_health(
                    text_input, model, tokenizer, label_encoder, stop_words
                )
                
                # Display results
                st.subheader("Analysis Results")
                
                # Show prediction
                st.markdown(f"**Predicted Status:** {predicted_class}")
                
                # Show probabilities
                import pandas as pd
                probs_df = pd.DataFrame({
                    'Category': label_encoder.classes_,
                    'Confidence': [f"{prob:.2%}" for prob in probabilities]
                })
                st.dataframe(probs_df, hide_index=True)
                
                # Create visualization
                import plotly.express as px
                fig = px.bar(
                    probs_df,
                    x='Category',
                    y=[float(p.strip('%'))/100 for p in probs_df['Confidence']],
                    title='Confidence Distribution Across Categories'
                )
                fig.update_layout(
                    yaxis_title='Confidence',
                    xaxis_title='Category',
                    xaxis_tickangle=45
                )
                st.plotly_chart(fig)
                
                # Display disclaimer
                st.markdown("""
                ---
                **Important:** This tool is for demonstration purposes only. If you're experiencing mental health concerns, please consult with a qualified healthcare professional.
                
                Crisis Resources:
                - National Crisis Hotline (US): 988
                - Crisis Text Line: Text HOME to 741741
                """)
                
            except Exception as e:
                st.error(f"An error occurred during analysis: {str(e)}")

if __name__ == "__main__":
    main()