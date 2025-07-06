import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re

# Set page config
st.set_page_config(
    page_title="Fake News Detection Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .user-message {
        color : black;
        background-color: #e3f2fd;
        padding: 0.8rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #2196f3;
    }
    .bot-message {
        color : black;    
        background-color: #f3e5f5;
        padding: 0.8rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #9c27b0;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-weight: bold;
    }
    .fake-news {
        background-color: #ffebee;
        color: #c62828;
        border: 2px solid #f44336;
    }
    .true-news {
        background-color: #e8f5e8;
        color: #2e7d32;
        border: 2px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

class FakeNewsPredictor:
    """Class for making predictions with trained model"""
    
    def __init__(self, model_path='./saved_model'):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.eval()
            self.model_loaded = True
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            self.model_loaded = False
    
    def clean_text(self, text):
        """Clean and preprocess text data without NLTK"""
        if pd.isna(text):
            return ""
        
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        tokens = text.split()
        filtered_tokens = [word for word in tokens if word not in ENGLISH_STOP_WORDS]
        return ' '.join(filtered_tokens)
    
    def predict(self, text, max_length=128):
        if not self.model_loaded:
            return {
                'prediction': 'Error',
                'confidence': 0.0,
                'probabilities': {'fake': 0.0, 'true': 0.0},
                'error': 'Model not loaded'
            }

        cleaned_text = self.clean_text(text)
        if not cleaned_text:
            return {
                'prediction': 'Error',
                'confidence': 0.0,
                'probabilities': {'fake': 0.0, 'true': 0.0},
                'error': 'Empty text after cleaning'
            }

        inputs = self.tokenizer(
            cleaned_text,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = torch.max(probabilities).item()

        label = "True" if predicted_class == 1 else "Fake"
        return {
            'prediction': label,
            'confidence': confidence,
            'probabilities': {
                'fake': probabilities[0][0].item(),
                'true': probabilities[0][1].item()
            }
        }

# Session state initialization
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'prediction_count' not in st.session_state:
    st.session_state.prediction_count = 0

@st.cache_resource
def load_model():
    return FakeNewsPredictor()

# Header
st.markdown('<h1 class="main-header">🤖 Fake News Detection Chatbot</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📊 Statistics")
    st.metric("Total Predictions", st.session_state.prediction_count)

    st.header("ℹ️ About")
    st.write("""
    This chatbot uses a fine-tuned DistilBERT model to detect fake news.
    Paste or type a news article and get an instant prediction!
    """)

    st.header("🔧 Model Info")
    st.write("""
    - Model: DistilBERT
    - Task: Binary Classification
    - Labels: Fake vs True
    """)

    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.prediction_count = 0
        st.rerun()

# Main layout
col1, col2 = st.columns([2, 1])

with col1:
    st.header("💬 Chat with the Bot")

    if st.session_state.chat_history:
        for user_msg, bot_msg, result in st.session_state.chat_history:
            st.markdown(f'<div class="user-message"><strong>You:</strong> {user_msg}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="bot-message"><strong>Bot:</strong> {bot_msg}</div>', unsafe_allow_html=True)

            if result and 'error' not in result:
                box_class = "fake-news" if result['prediction'] == 'Fake' else "true-news"
                st.markdown(f'''
                <div class="prediction-box {box_class}">
                    <strong>Prediction:</strong> {result['prediction']} News<br>
                    <strong>Confidence:</strong> {result['confidence']:.1%}
                </div>
                ''', unsafe_allow_html=True)

    with st.form("news_input", clear_on_submit=True):
        user_input = st.text_area("Enter news text to analyze:", height=150)
        col_a, col_b = st.columns([1, 4])
        with col_a:
            submit_button = st.form_submit_button("🔍 Analyze", use_container_width=True)
        with col_b:
            example_button = st.form_submit_button("📰 Try Example", use_container_width=True)

    if example_button:
        user_input = (
            "Scientists have discovered a new planet in our solar system that is twice the size of Earth. "
            "NASA officials confirm that this planet, named Kepler-442b, shows signs of water and potentially life."
        )
        submit_button = True

    if submit_button and user_input:
        if len(user_input.strip()) < 20:
            st.error("Please enter a longer news text (at least 20 characters).")
        else:
            predictor = load_model()
            with st.spinner("Analyzing news..."):
                result = predictor.predict(user_input)

            if 'error' in result:
                bot_response = f"Sorry, I encountered an error: {result['error']}"
            else:
                confidence_text = f"{result['confidence']:.1%}"
                if result['prediction'] == 'Fake':
                    bot_response = f"🚨 This appears to be **FAKE NEWS** with {confidence_text} confidence."
                else:
                    bot_response = f"✅ This appears to be **TRUE NEWS** with {confidence_text} confidence."

            st.session_state.chat_history.append((user_input, bot_response, result))
            st.session_state.prediction_count += 1
            st.rerun()

with col2:
    st.header("📈 Analytics")

    if st.session_state.chat_history:
        predictions = []
        confidences = []

        for _, _, result in st.session_state.chat_history:
            if result and 'error' not in result:
                predictions.append(result['prediction'])
                confidences.append(result['confidence'])

        if predictions:
            st.subheader("Prediction Distribution")
            pred_counts = pd.Series(predictions).value_counts()
            fig_pie = px.pie(values=pred_counts.values, names=pred_counts.index,
                             color_discrete_map={'Fake': '#ff6b6b', 'True': '#51cf66'})
            fig_pie.update_layout(height=300)
            st.plotly_chart(fig_pie, use_container_width=True)

            st.subheader("Confidence Scores")
            fig_bar = px.bar(
                x=list(range(1, len(confidences) + 1)),
                y=confidences,
                color=predictions,
                color_discrete_map={'Fake': '#ff6b6b', 'True': '#51cf66'},
                labels={'x': 'Prediction #', 'y': 'Confidence'}
            )
            fig_bar.update_layout(height=300)
            st.plotly_chart(fig_bar, use_container_width=True)

            latest_result = st.session_state.chat_history[-1][2]
            if latest_result and 'error' not in latest_result:
                st.subheader("Latest Prediction Details")
                st.write(f"**Prediction:** {latest_result['prediction']}")
                st.write(f"**Confidence:** {latest_result['confidence']:.1%}")
                st.write("**Probability Breakdown:**")
                st.write(f"- Fake: {latest_result['probabilities']['fake']:.1%}")
                st.write(f"- True: {latest_result['probabilities']['true']:.1%}")
    else:
        st.info("No predictions yet. Start by entering some news text!")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>🤖 Powered by DistilBERT | Built with Streamlit</p>
        <p><em>Always verify important news from multiple reliable sources!</em></p>
    </div>
    """,
    unsafe_allow_html=True
)
