# src/data_preprocessing.py
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

# Load dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df[['text', 'rating_category']]  # Use 'rating_category' as the label
    return df

# Tokenize the reviews
def tokenize_data(df, tokenizer, max_length=128):
    encodings = tokenizer(df['text'].tolist(), padding=True, truncation=True, max_length=max_length)
    return encodings

# Split data into training and validation sets
def split_data(df, test_size=0.2, random_state=42):
    return train_test_split(df, test_size=test_size, random_state=random_state)
