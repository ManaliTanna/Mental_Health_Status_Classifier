import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Mental Health Status Classifier",
    page_icon="🧠",
    layout="centered"
)

@st.cache_resource
def load_model():
    """Load the BERT model and tokenizer."""
    try:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=7,
        )
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def preprocess_text(text):
    """Preprocess the input text."""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and extra whitespace
    import re
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join(text.split())
    return text

def predict_mental_health(text, tokenizer, model):
    """Make prediction on input text."""
    # Preprocess the text
    text = preprocess_text(text)
    
    # Prepare the text
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=-1).item()
    
    return predicted_class, predictions[0].tolist()

def main():
    st.title("🧠 Mental Health Status Classifier")
    st.markdown("""
    This application analyzes text to identify potential mental health status indicators.
    
    **Note: This is a demonstration tool and should not be used as a substitute for professional medical advice.**
    """)
    
    # Load model
    tokenizer, model = load_model()
    if model is None:
        return
    
    model.eval()
    
    # Text input
    text_input = st.text_area(
        "Enter text to analyze:",
        height=150,
        placeholder="Type or paste text here..."
    )
    
    # Class labels from your paper
    class_labels = [
        "Anxiety",
        "Bipolar",
        "Depression",
        "Normal",
        "Personality Disorder",
        "Stress",
        "Suicidal"
    ]
    
    if st.button("Analyze"):
        if not text_input:
            st.warning("Please enter some text to analyze.")
            return
            
        with st.spinner("Analyzing..."):
            try:
                predicted_class, probabilities = predict_mental_health(text_input, tokenizer, model)
                
                # Display results
                st.subheader("Results")
                
                # Display prediction
                st.markdown(f"**Primary Classification:** {class_labels[predicted_class]}")
                
                # Display probabilities
                import pandas as pd
                probs_df = pd.DataFrame({
                    'Category': class_labels,
                    'Confidence': [f"{prob:.2%}" for prob in probabilities]
                })
                st.dataframe(probs_df, hide_index=True)
                
                # Display disclaimer
                st.markdown("""
                ---
                **Important:** This is a demonstration only. Please consult healthcare professionals for actual mental health concerns.
                
                Crisis Resources:
                - National Crisis Hotline (US): 988
                - Crisis Text Line: Text HOME to 741741
                """)
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()