# Mental Health Status Classifier

Deep Learning Project NYU

## Overview
This project implements a finetuned BERT-based model for detecting mental health status from text input. The model can classify text into seven categories: Anxiety, Bipolar, Depression, Normal, Personality Disorder, Stress, and Suicidal. Built using the Transformers library and deployed as a Streamlit web application, this tool demonstrates the potential of AI in mental health screening while emphasizing its role as a supplementary tool rather than a replacement for professional medical advice.

## Features
- Real-time text analysis using BERT
- Multi-class classification across 7 mental health categories
- User-friendly web interface
- Visualization of prediction probabilities
- Preprocessing pipeline for text input
- Important disclaimers and crisis resources

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup
1. Clone the repository:
```bash
git clone [your-repository-url]
cd mental-health-app
```

2. Create and activate a virtual environment:
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On MacOS/Linux:
source .venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to `http://localhost:8501`

3. Enter text in the input field and click "Analyze" to get predictions

## Model Details

The classifier uses a fine-tuned BERT model with the following specifications:
- Base Model: BERT-base-uncased
- Number of Labels: 7
- Training Parameters:
  - Learning Rate: 2×10^-5
  - Batch Size: 16
  - Number of Epochs: 5
  - Weight Decay: 0.01
  - Optimizer: AdamW
  - Scheduler: Linear with warmup

## Performance Metrics

| Class             | Precision | Recall | F1-Score |
|-------------------|-----------|---------|-----------|
| Anxiety          | 0.97      | 0.98    | 0.97      |
| Bipolar          | 0.98      | 0.99    | 0.98      |
| Depression       | 0.85      | 0.72    | 0.78      |
| Normal           | 0.95      | 0.91    | 0.93      |
| PD               | 0.99      | 1.00    | 0.99      |
| Stress           | 0.93      | 1.00    | 0.96      |
| Suicidal         | 0.79      | 0.85    | 0.82      |

Overall Accuracy: 92%
