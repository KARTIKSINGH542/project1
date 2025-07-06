"""
Main script to run the complete fake news detection pipeline
"""

import os
import sys
import subprocess
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import re
import warnings
warnings.filterwarnings('ignore')

# Simple English stopwords list
STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 
    'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 
    'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 
    'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 
    'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 
    'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after', 
    'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 
    'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will', 'just', 
    'don', 'should', 'now'
}

def clean_text(text):
    """Clean and preprocess text without NLTK"""
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Simple tokenization and stopword removal
    tokens = text.split()
    tokens = [token for token in tokens if token not in STOPWORDS and len(token) > 2]
    
    return ' '.join(tokens)

def preprocess_data(true_path, fake_path, sample_size=None):
    """
    Preprocess the fake news dataset
    
    Args:
        true_path: Path to true news CSV
        fake_path: Path to fake news CSV
        sample_size: Number of samples to use from each class (None for all)
    
    Returns:
        train_df, val_df, test_df: Processed dataframes
    """
    
    print("🔄 Loading raw data...")
    
    # Load datasets
    try:
        true_df = pd.read_csv(true_path)
        fake_df = pd.read_csv(fake_path)
        print(f"✅ Loaded {len(true_df)} true articles and {len(fake_df)} fake articles")
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find data files. Please ensure {true_path} and {fake_path} exist.")
        raise e
    
    # Sample data if specified
    if sample_size:
        true_df = true_df.sample(n=min(sample_size, len(true_df)), random_state=42)
        fake_df = fake_df.sample(n=min(sample_size, len(fake_df)), random_state=42)
        print(f"📊 Sampled {len(true_df)} true and {len(fake_df)} fake articles")
    
    # Add labels
    true_df['label'] = 1  # True news
    fake_df['label'] = 0  # Fake news
    
    # Combine dataframes
    df = pd.concat([true_df, fake_df], ignore_index=True)
    
    # Create text column (combine title and text if both exist)
    if 'title' in df.columns and 'text' in df.columns:
        df['raw_text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
    elif 'title' in df.columns:
        df['raw_text'] = df['title'].fillna('')
    elif 'text' in df.columns:
        df['raw_text'] = df['text'].fillna('')
    else:
        # Try to find any text column
        text_columns = [col for col in df.columns if 'text' in col.lower() or 'content' in col.lower()]
        if text_columns:
            df['raw_text'] = df[text_columns[0]].fillna('')
        else:
            raise ValueError("No text column found in the dataset")
    
    # Remove empty texts
    df = df[df['raw_text'].str.strip() != '']
    
    print("🧹 Cleaning text data...")
    
    # Clean text
    df['cleaned_text'] = df['raw_text'].apply(clean_text)
    
    # Remove empty cleaned texts
    df = df[df['cleaned_text'].str.strip() != '']
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['cleaned_text'])
    
    print(f"✅ Final dataset size: {len(df)} articles")
    
    # Split the data
    print("📊 Splitting data...")
    
    # First split: train + temp (80%) and test (20%)
    train_temp, test_df = train_test_split(
        df, 
        test_size=0.2, 
        random_state=42, 
        stratify=df['label']
    )
    
    # Second split: train (60%) and validation (20%)
    train_df, val_df = train_test_split(
        train_temp, 
        test_size=0.25,  # 0.25 * 0.8 = 0.2 of total
        random_state=42, 
        stratify=train_temp['label']
    )
    
    print(f"📈 Data split completed:")
    print(f"  - Train: {len(train_df)} samples")
    print(f"  - Validation: {len(val_df)} samples")
    print(f"  - Test: {len(test_df)} samples")
    
    # Save processed data
    print("💾 Saving processed data...")
    train_df.to_csv('train_data.csv', index=False)
    val_df.to_csv('val_data.csv', index=False)
    test_df.to_csv('test_data.csv', index=False)
    
    print("✅ Data preprocessing completed!")
    
    return train_df, val_df, test_df

def train_model(model_name='distilbert-base-uncased'):
    """Import and run the training function"""
    from model_training import train_model as train_func
    return train_func(model_name)

def run_pipeline():
    """
    Run the complete fake news detection pipeline
    """
    print("="*50)
    print("FAKE NEWS DETECTION PIPELINE")
    print("="*50)
    
    # Step 1: Check if CSV files exist
    if not os.path.exists('true.csv') or not os.path.exists('fake.csv'):
        print("❌ Error: true.csv and fake.csv files not found!")
        print("Please ensure you have the dataset files in the current directory.")
        print("You can download the dataset from: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset")
        return False
    
    # Step 2: Data preprocessing
    print("\n🔄 Step 1: Data Preprocessing")
    try:
        train_df, val_df, test_df = preprocess_data('true.csv', 'fake.csv', sample_size=1000)
        print("✅ Data preprocessing completed successfully!")
    except Exception as e:
        print(f"❌ Error in data preprocessing: {str(e)}")
        return False
    
    # Step 3: Model training
    print("\n🔄 Step 2: Model Training")
    try:
        model, tokenizer, results = train_model('distilbert-base-uncased')
        print("✅ Model training completed successfully!")
        print(f"📊 Test Results: {results}")
    except Exception as e:
        print(f"❌ Error in model training: {str(e)}")
        return False
    
    # Step 4: Verify model files
    if os.path.exists('./saved_model') and os.path.isdir('./saved_model'):
        print("✅ Model saved successfully!")
    else:
        print("❌ Error: Model not saved properly!")
        return False
    
    print("\n" + "="*50)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*50)
    print("\n💡 You can now run the Streamlit app with:")
    print("👉 streamlit run streamlit_app.py")
    
    return True

def setup_environment():
    """
    Setup the environment and install dependencies
    """
    print("🛠️ Setting up environment...")
    
    required_packages = [
        'torch',
        'transformers',
        'pandas',
        'numpy',
        'scikit-learn',
        'streamlit',
        'plotly',
        'datasets',
        'accelerate'
    ]
    
    for package in required_packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package, '--quiet'])
            print(f"✅ Installed: {package}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install: {package}")
    
    print("✅ Environment setup complete!\n")

if __name__ == "__main__":
    print("🚀 Starting Fake News Detection Pipeline...")
    setup_environment()
    
    if run_pipeline():
        print("\n🎉 All done! Your fake news detection model is ready to use.")
    else:
        print("\n❌ Pipeline failed. Please check the errors above.")