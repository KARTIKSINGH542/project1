import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import warnings
warnings.filterwarnings('ignore')

def clean_text(text):
    """Clean and preprocess text data"""
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize and remove stopwords using scikit-learn's list
    tokens = text.split()
    filtered_tokens = [word for word in tokens if word not in ENGLISH_STOP_WORDS]
    
    return ' '.join(filtered_tokens)

def preprocess_data(true_csv_path, fake_csv_path, sample_size=100):
    """
    Preprocess the fake news dataset
    """
    # Load datasets
    print("Loading datasets...")
    true_df = pd.read_csv(true_csv_path)
    fake_df = pd.read_csv(fake_csv_path)

    true_df = true_df.head(sample_size)
    fake_df = fake_df.head(sample_size)

    print(f"True news samples: {len(true_df)}")
    print(f"Fake news samples: {len(fake_df)}")

    def combine_text_columns(df):
        text_columns = []
        if 'title' in df.columns:
            text_columns.append('title')
        if 'text' in df.columns:
            text_columns.append('text')
        if 'subject' in df.columns:
            text_columns.append('subject')
        
        if len(text_columns) > 1:
            df['combined_text'] = df[text_columns].fillna('').agg(' '.join, axis=1)
        elif len(text_columns) == 1:
            df['combined_text'] = df[text_columns[0]].fillna('')
        else:
            text_col = df.select_dtypes(include=['object']).columns[0]
            df['combined_text'] = df[text_col].fillna('')
        
        return df

    # Combine text columns
    true_df = combine_text_columns(true_df)
    fake_df = combine_text_columns(fake_df)

    # Add labels
    true_df['label'] = 1
    fake_df['label'] = 0

    # Combine and clean
    combined_df = pd.concat([true_df[['combined_text', 'label']], 
                             fake_df[['combined_text', 'label']]], ignore_index=True)

    print("Cleaning text data...")
    combined_df['cleaned_text'] = combined_df['combined_text'].apply(clean_text)
    combined_df = combined_df[combined_df['cleaned_text'].str.len() > 0]

    # Shuffle
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split
    print("Splitting dataset...")
    train_df, temp_df = train_test_split(combined_df, test_size=0.2, random_state=42, stratify=combined_df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

    print(f"Training set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    print(f"Test set: {len(test_df)} samples")

    # Save
    train_df.to_csv('train_data.csv', index=False)
    val_df.to_csv('val_data.csv', index=False)
    test_df.to_csv('test_data.csv', index=False)

    print("Data preprocessing completed!")
    print("Label distribution in training set:")
    print(train_df['label'].value_counts())

    return train_df, val_df, test_df

if __name__ == "__main__":
    train_df, val_df, test_df = preprocess_data('true.csv', 'fake.csv', sample_size=100)
    
    print("\nSample processed data:")
    print(train_df.head())
