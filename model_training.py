import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pickle
import warnings
warnings.filterwarnings('ignore')

class FakeNewsDataset(Dataset):
    """Custom dataset for fake news detection"""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    accuracy = accuracy_score(labels, predictions)
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_model(model_name='distilbert-base-uncased'):
    """
    Train the fake news detection model
    
    Args:
        model_name: Pre-trained model name from Hugging Face
    """
    
    # Load processed data
    print("Loading processed data...")
    try:
        train_df = pd.read_csv('train_data.csv')
        val_df = pd.read_csv('val_data.csv')
        test_df = pd.read_csv('test_data.csv')
        print(f"✅ Data loaded - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    except FileNotFoundError as e:
        print(f"❌ Error: Required CSV files not found. Please run data preprocessing first.")
        raise e
    
    # Initialize tokenizer and model
    print(f"Loading {model_name} model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        problem_type="single_label_classification"
    )
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = FakeNewsDataset(
        train_df['cleaned_text'].tolist(),
        train_df['label'].tolist(),
        tokenizer
    )
    
    val_dataset = FakeNewsDataset(
        val_df['cleaned_text'].tolist(),
        val_df['label'].tolist(),
        tokenizer
    )
    
    test_dataset = FakeNewsDataset(
        test_df['cleaned_text'].tolist(),
        test_df['label'].tolist(),
        tokenizer
    )
    
    # Training arguments - FIXED VERSION
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,  # Reduced batch size
        per_device_eval_batch_size=8,
        logging_steps=10,
        
        # Required for EarlyStoppingCallback
        eval_strategy="epoch",
        save_strategy="epoch",
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        load_best_model_at_end=True,
        
        # Additional settings
        save_total_limit=2,
        report_to=None,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # Train model
    print("Starting training...")
    trainer.train()
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_results = trainer.evaluate(test_dataset)
    print(f"Test Results: {test_results}")
    
    # Save model and tokenizer
    print("Saving model and tokenizer...")
    model.save_pretrained('./saved_model')
    tokenizer.save_pretrained('./saved_model')
    
    # Save training history
    with open('training_results.pkl', 'wb') as f:
        pickle.dump(test_results, f)
    
    print("Model training completed!")
    return model, tokenizer, test_results

class FakeNewsPredictor:
    """Class for making predictions with trained model"""
    
    def __init__(self, model_path='./saved_model'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def predict(self, text, max_length=128):
        """
        Predict if news text is fake or true
        
        Args:
            text: Input news text
            max_length: Maximum sequence length
        
        Returns:
            Dictionary with prediction and confidence
        """
        # Tokenize input
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = torch.max(probabilities).item()
        
        # Convert to readable format
        label = "True" if predicted_class == 1 else "Fake"
        
        return {
            'prediction': label,
            'confidence': confidence,
            'probabilities': {
                'fake': probabilities[0][0].item(),
                'true': probabilities[0][1].item()
            }
        }

if __name__ == "__main__":
    # Train the model
    model, tokenizer, results = train_model()
    
    # Test the predictor
    predictor = FakeNewsPredictor()
    
    # Example predictions
    test_texts = [
        "Scientists discover new planet in solar system",
        "Breaking: Aliens land on Earth, government confirms contact"
    ]
    
    for text in test_texts:
        result = predictor.predict(text)
        print(f"Text: {text}")
        print(f"Prediction: {result['prediction']} (Confidence: {result['confidence']:.2f})")
        print()