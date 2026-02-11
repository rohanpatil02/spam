# app.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
import os

# Page configuration
st.set_page_config(
    page_title="SpamShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    
    .prediction-box {
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.3rem;
        font-weight: bold;
        animation: fadeIn 0.5s ease-in;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    .spam {
        background: linear-gradient(135deg, #FFE5E5, #FFB8B8);
        color: #9B1C1C;
        border-left: 8px solid #DC2626;
    }
    
    .ham {
        background: linear-gradient(135deg, #D1FAE5, #A7F3D0);
        color: #065F46;
        border-left: 8px solid #10B981;
    }
    
    .stButton>button {
        background: linear-gradient(45deg, #3B82F6, #1D4ED8);
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.75rem 2.5rem;
        border-radius: 10px;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

def load_complete_model():
    """Load the complete model with vectorizer"""
    try:
        # Try to load the complete model
        model_data = joblib.load('spam_detector_complete.joblib')
        
        model = model_data['model']
        vectorizer = model_data['vectorizer']
        feature_count = model_data['feature_count']
        accuracy = model_data.get('accuracy', 'Unknown')
        
        return {
            'model': model,
            'vectorizer': vectorizer,
            'feature_count': feature_count,
            'accuracy': accuracy,
            'status': 'success',
            'message': f'✅ Model loaded! Features: {feature_count}, Accuracy: {accuracy:.2%}' if isinstance(accuracy, float) else f'✅ Model loaded! Features: {feature_count}'
        }
    except FileNotFoundError:
        # Fallback: try to load separate files
        try:
            model = joblib.load('spam_email_model.joblib')
            
            # Try to load vectorizer separately
            if os.path.exists('count_vectorizer.joblib'):
                vectorizer = joblib.load('count_vectorizer.joblib')
                feature_count = len(vectorizer.get_feature_names_out())
                return {
                    'model': model,
                    'vectorizer': vectorizer,
                    'feature_count': feature_count,
                    'accuracy': 'Unknown',
                    'status': 'success',
                    'message': f'✅ Model loaded from separate files. Features: {feature_count}'
                }
            else:
                return {
                    'status': 'error',
                    'message': '❌ Vectorizer not found. Please run save_complete_model.py first.'
                }
        except Exception as e:
            return {
                'status': 'error',
                'message': f'❌ Error loading model: {str(e)}'
            }
    except Exception as e:
        return {
            'status': 'error',
            'message': f'❌ Error: {str(e)}'
        }

def predict_email(model_data, email_text):
    """Make prediction on email text using the loaded model and vectorizer"""
    try:
        model = model_data['model']
        vectorizer = model_data['vectorizer']
        
        # Transform the text using the SAME vectorizer used during training
        email_vectorized = vectorizer.transform([email_text])
        
        # Make prediction
        prediction = model.predict(email_vectorized)[0]
        
        # Get prediction probabilities
        probabilities = model.predict_proba(email_vectorized)[0]
        confidence = max(probabilities) * 100
        
        # Get feature importance for top words
        feature_names = vectorizer.get_feature_names_out()
        X = email_vectorized.toarray()[0]
        
        # Get feature log probabilities
        if prediction == 1:  # spam
            feature_probs = model.feature_log_prob_[1]
        else:  # ham
            feature_probs = model.feature_log_prob_[0]
        
        # Calculate word contributions
        contributions = X * feature_probs
        nonzero_indices = np.where(X > 0)[0]
        
        if len(nonzero_indices) > 0:
            top_indices = nonzero_indices[np.argsort(contributions[nonzero_indices])[-5:][::-1]]
            top_words = [(feature_names[i], contributions[i]) for i in top_indices]
        else:
            top_words = []
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': probabilities,
            'top_words': top_words,
            'features_used': X.sum(),  # Number of words that matched vocabulary
            'total_features': len(feature_names)
        }
    except Exception as e:
        return {
            'error': str(e),
            'prediction': None,
            'confidence': None
        }

def analyze_email_features(email_text, vectorizer):
    """Analyze which features from the model are present in the email"""
    try:
        # Transform email
        X = vectorizer.transform([email_text]).toarray()[0]
        feature_names = vectorizer.get_feature_names_out()
        
        # Get present features
        present_indices = np.where(X > 0)[0]
        present_features = [(feature_names[i], X[i]) for i in present_indices]
        
        return {
            'total_features_present': len(present_features),
            'present_features': present_features[:10],  # First 10
            'feature_vector': X
        }
    except:
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🛡️ SpamShield AI</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #6B7280; margin-bottom: 2rem;">Professional Email Spam Detection System</p>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'model_data' not in st.session_state:
        st.session_state.model_data = None
        st.session_state.loaded = False
        st.session_state.history = []
    
    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3067/3067256.png", width=80)
        st.markdown("### Model Management")
        
        # Model loading section
        st.markdown("---")
        st.markdown("#### 🔧 Load Model")
        
        model_option = st.radio(
            "Choose model file:",
            ["Complete Model (Recommended)", "Separate Files"]
        )
        
        if model_option == "Complete Model (Recommended)":
            model_path = "spam_detector_complete.joblib"
        else:
            model_path = "spam_email_model.joblib"
        
        if st.button("🚀 Load Detection Model", use_container_width=True):
            with st.spinner("Loading model and vectorizer..."):
                result = load_complete_model()
                
                if result['status'] == 'success':
                    st.session_state.model_data = result
                    st.session_state.loaded = True
                    st.success(result['message'])
                    
                    # Show model info
                    with st.expander("📊 Model Details"):
                        st.write(f"**Model Type:** {type(result['model']).__name__}")
                        st.write(f"**Features:** {result['feature_count']}")
                        st.write(f"**Vocabulary Size:** {len(result['vectorizer'].get_feature_names_out())}")
                        if isinstance(result['accuracy'], float):
                            st.write(f"**Accuracy:** {result['accuracy']:.2%}")
                else:
                    st.error(result['message'])
        
        st.markdown("---")
        
        # Sample emails
        if st.session_state.loaded:
            st.markdown("#### 📋 Try Samples")
            sample = st.selectbox(
                "Select a sample email:",
                ["Choose sample...", "Legitimate Work Email", "Spam: Lottery Win", 
                 "Spam: Urgent Bank Alert", "Legitimate Newsletter", "Spam: Pharmacy"]
            )
            
            samples = {
                "Legitimate Work Email": """Hi Team,

Following up on yesterday's meeting about the Q3 roadmap. 
Please find attached the presentation slides and action items.

Let's schedule a follow-up for next Tuesday at 2 PM.

Best regards,
Alex
Project Manager""",
                
                "Spam: Lottery Win": """CONGRATULATIONS! YOU'VE WON $1,000,000!

URGENT: Claim your prize within 24 hours!
Click here: http://bit.ly/win-now-2024

This is a LIMITED TIME OFFER!!! Don't miss out!""",
                
                "Spam: Urgent Bank Alert": """SECURITY ALERT: Unusual activity detected!

Dear Customer,
Your account has been temporarily suspended due to suspicious login attempts.

VERIFY YOUR IDENTITY NOW: http://secure-bank-verify.com/login

This is your FINAL WARNING!""",
                
                "Legitimate Newsletter": """Hello Subscriber,

Welcome to our monthly tech newsletter! This edition includes:
1. Latest AI developments
2. Upcoming webinars
3. Community highlights

To unsubscribe, click here: [unsubscribe link]

Best,
The TechTeam""",
                
                "Spam: Pharmacy": """LIMITED TIME OFFER! 80% OFF!

Get prescription medications without doctor visit!
Viagra, Cialis, Xanax - ALL AVAILABLE!

Order now: http://online-pharmacy-24-7.com

FREE SHIPPING! DISCREET PACKAGING!"""
            }
            
            if sample in samples:
                st.session_state.sample_email = samples[sample]
                st.info(f"Sample '{sample}' loaded. Click analyze to test.")
        
        st.markdown("---")
        st.caption("Made with ❤️ using Streamlit")
    
    # Main content
    if not st.session_state.loaded:
        # Welcome screen
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 Welcome to SpamShield AI")
            st.markdown("""
            This is a professional email spam detection system using:
            
            **🤖 Model:** Multinomial Naive Bayes
            **📊 Features:** 7,788 unique words
            **🎯 Accuracy:** ~98% on test data
            
            **To get started:**
            1. Click **'Load Detection Model'** in the sidebar
            2. Enter or paste an email
            3. Get instant spam analysis
            
            **Features include:**
            • Real-time prediction with confidence scores
            • Word contribution analysis
            • Batch email processing
            • History tracking
            """)
        
        with col2:
            st.markdown("### 📝 Quick Start Guide")
            st.markdown("""
            **If you see feature mismatch error:**
            
            1. **Run the save script:**
            ```bash
            python save_complete_model.py
            ```
            
            2. **This creates:** `spam_detector_complete.joblib`
            
            3. **Then load in sidebar**
            
            **Requirements:**
            - Original `spam.csv` file
            - Python with scikit-learn
            - Run once to save properly
            """)
            
            st.warning("⚠️ Please load the model first to start analysis!")
            
            if st.button("🔄 Check Available Models"):
                import os
                files = os.listdir('.')
                model_files = [f for f in files if 'joblib' in f or 'pkl' in f]
                
                if model_files:
                    st.success("Found model files:")
                    for f in model_files:
                        size = os.path.getsize(f) / 1024
                        st.write(f"• {f} ({size:.1f} KB)")
                else:
                    st.error("No model files found!")
    
    else:
        # Main interface when model is loaded
        st.success(f"✅ Model ready! Features: {st.session_state.model_data['feature_count']}")
        
        # Input section
        st.markdown("### ✉️ Email Analysis")
        
        # Email input
        email_text = st.text_area(
            "Paste email content here:",
            height=200,
            value=st.session_state.get('sample_email', ''),
            placeholder="Enter the email text to analyze for spam..."
        )
        
        # Stats about input
        if email_text:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Words", len(email_text.split()))
            with col2:
                st.metric("Characters", len(email_text))
            with col3:
                links = len(re.findall(r'http[s]?://', email_text))
                st.metric("Links", links)
            with col4:
                exclamations = email_text.count('!')
                st.metric("!", exclamations)
        
        # Analysis options
        col_opt1, col_opt2, col_opt3 = st.columns(3)
        with col_opt1:
            show_details = st.checkbox("Show detailed analysis", value=True)
        with col_opt2:
            show_features = st.checkbox("Show matched features", value=False)
        with col_opt3:
            add_history = st.checkbox("Add to history", value=True)
        
        # Analyze button
        if st.button("🔬 Analyze Email for Spam", type="primary", use_container_width=True):
            if not email_text.strip():
                st.warning("Please enter some email content!")
            else:
                with st.spinner("Analyzing email content..."):
                    # Make prediction
                    result = predict_email(st.session_state.model_data, email_text)
                    
                    if 'error' in result:
                        st.error(f"Prediction error: {result['error']}")
                    else:
                        # Display results
                        st.markdown("---")
                        st.markdown("## 📊 Analysis Results")
                        
                        # Prediction with confidence
                        prediction = result['prediction']
                        confidence = result['confidence']
                        is_spam = prediction == 1
                        
                        # Big result card
                        if is_spam:
                            st.markdown(
                                f'<div class="prediction-box spam">'
                                f'🚨 <h2>SPAM DETECTED!</h2>'
                                f'<h3>Confidence: {confidence:.1f}%</h3>'
                                f'<p>This email is highly likely to be spam.</p>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                f'<div class="prediction-box ham">'
                                f'✅ <h2>LEGITIMATE EMAIL</h2>'
                                f'<h3>Confidence: {confidence:.1f}%</h3>'
                                f'<p>This email appears to be legitimate.</p>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                        
                        # Probability breakdown
                        if show_details:
                            st.markdown("### 📈 Probability Breakdown")
                            
                            prob_data = pd.DataFrame({
                                'Class': ['Legitimate (Ham)', 'Spam'],
                                'Probability': [result['probabilities'][0] * 100, result['probabilities'][1] * 100]
                            })
                            
                            fig = px.bar(prob_data, x='Class', y='Probability',
                                        color='Class',
                                        color_discrete_map={
                                            'Legitimate (Ham)': '#10B981',
                                            'Spam': '#DC2626'
                                        },
                                        text='Probability')
                            fig.update_traces(texttemplate='%{y:.2f}%', textposition='outside')
                            fig.update_layout(
                                showlegend=False,
                                yaxis_title="Probability (%)",
                                height=400
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Feature analysis
                            if result['top_words']:
                                st.markdown("### 🔍 Top Contributing Words")
                                words_df = pd.DataFrame(
                                    result['top_words'],
                                    columns=['Word', 'Contribution']
                                )
                                st.dataframe(words_df, use_container_width=True)
                            
                            # Feature matching info
                            st.info(f"📊 **Feature Matching:** {result['features_used']} words matched the model's vocabulary out of {result['total_features']} total features")
                        
                        # Show matched features
                        if show_features:
                            feature_analysis = analyze_email_features(email_text, st.session_state.model_data['vectorizer'])
                            if feature_analysis:
                                st.markdown("### 🎯 Matched Features")
                                st.write(f"**Total features present:** {feature_analysis['total_features_present']}")
                                
                                if feature_analysis['present_features']:
                                    features_df = pd.DataFrame(
                                        feature_analysis['present_features'],
                                        columns=['Word', 'Count']
                                    )
                                    st.dataframe(features_df.head(20), use_container_width=True)
                        
                        # Add to history
                        if add_history:
                            st.session_state.history.append({
                                'email': email_text[:100] + "..." if len(email_text) > 100 else email_text,
                                'prediction': "SPAM" if is_spam else "HAM",
                                'confidence': confidence,
                                'time': datetime.now().strftime("%H:%M:%S")
                            })
                        
                        # Tips based on prediction
                        st.markdown("---")
                        st.markdown("### 💡 Recommendations")
                        if is_spam:
                            st.warning("""
                            **⚠️ This email is likely spam. Recommendations:**
                            - Do not click any links
                            - Do not download attachments
                            - Do not reply with personal information
                            - Mark as spam in your email client
                            - Delete the email
                            """)
                        else:
                            st.success("""
                            **✅ This email appears legitimate. Recommendations:**
                            - Check sender address matches expected contact
                            - Verify links before clicking (hover to see URL)
                            - Be cautious with attachments
                            - If unsure, contact sender through other channels
                            """)
        
        # History section
        if st.session_state.history:
            st.markdown("---")
            st.markdown("### 📜 Analysis History")
            
            history_df = pd.DataFrame(st.session_state.history)
            st.dataframe(
                history_df.style.apply(
                    lambda x: ['background-color: #FEE2E2' if v == 'SPAM' else 'background-color: #D1FAE5' for v in x],
                    subset=['prediction']
                ),
                use_container_width=True
            )
            
            if st.button("Clear History"):
                st.session_state.history = []
                st.rerun()

if __name__ == "__main__":
    main()