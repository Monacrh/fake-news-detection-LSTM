# import streamlit as st
# import pickle
# import re
# import nltk
# from nltk.corpus import stopwords
# from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# # Download required NLTK data
# nltk.download('stopwords')

# # Initialize Indonesian stemmer
# factory = StemmerFactory()
# indo_stemmer = factory.create_stemmer()
# indonesian_stopwords = set(stopwords.words('indonesian'))

# # Load resources
# try:
#     vector_form = pickle.load(open('vector.pkl', 'rb'))
#     load_model = pickle.load(open('model.pkl', 'rb'))
#     encoder = pickle.load(open('encoder.pkl', 'rb'))  # Load label encoder
# except Exception as e:
#     st.error(f"Error loading model files: {str(e)}")
#     st.stop()

# def stemming(text):
#     # Preprocess Indonesian text
#     text = re.sub('[^a-zA-Z]', ' ', text)
#     text = text.lower()
#     words = text.split()
#     stemmed_words = [indo_stemmer.stem(word) for word in words if word not in indonesian_stopwords]
#     return ' '.join(stemmed_words)

# def fake_news(news):
#     try:
#         # Preprocess input
#         news = stemming(news)
#         input_data = [news]
        
#         # Vectorize
#         vector_form1 = vector_form.transform(input_data)
        
#         # Predict
#         prediction = load_model.predict(vector_form1)
        
#         # Decode prediction
#         return encoder.inverse_transform(prediction)[0]
#     except Exception as e:
#         st.error(f"Prediction error: {str(e)}")
#         return None

# # Streamlit UI
# st.title('Indonesian Fake News Classification App')
# st.subheader("Enter News Content")

# with st.form(key='news_form'):
#     sentence = st.text_area("M", "insert news text here", height=200)
#     submit_button = st.form_submit_button(label='Check Authenticity')

# if submit_button:
#     if len(sentence.strip()) < 10:
#         st.warning("Enter longer news text (minimum 10 characters)")
#     else:
#         with st.spinner('Analyzing news...'):
#             prediction = fake_news(sentence)
            
#         if prediction is not None:
#             if prediction == 'reliable':
#                 st.success('✅ Reliable News')
#                 st.balloons()
#             else:
#                 st.error('❌ Unreliable News')
#                 st.write("Be careful with this information and verify it through official sources")

# st.markdown("""
# ---
# **Catatan**: 
# - This application uses a machine learning model to predict the authenticity of news
# - Prediction results are probabilistic and not absolute certainty
# - Always verify information through official sources
# """)

# if __name__ == '__main__':
#     st.write("The application is ready to use!")

import streamlit as st
import pickle
import re
import nltk
import tensorflow as tf
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Initialize stemmer and stopwords
nltk.download('stopwords')
factory = StemmerFactory()
indo_stemmer = factory.create_stemmer()
indonesian_stopwords = set(nltk.corpus.stopwords.words('indonesian'))

# Load resources
try:
    # Load LSTM model and tokenizer
    model = tf.keras.models.load_model('fake_news_lstm.h5')
    with open('tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)
    
    # Load label encoder
    with open('vector.pkl', 'rb') as f:
        encoder = pickle.load(f)
except Exception as e:
    st.error(f"Error loading model files: {str(e)}")
    st.stop()

# Preprocessing function (must match training preprocessing)
def preprocess_text(text):
    # Clean text
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    
    # Tokenize and stem
    words = text.split()
    stemmed_words = [indo_stemmer.stem(word) for word in words if word not in indonesian_stopwords]
    return ' '.join(stemmed_words)

def predict_news(text):
    try:
        # Preprocess input
        processed_text = preprocess_text(text)
        
        # Convert to sequence
        sequence = tokenizer.texts_to_sequences([processed_text])
        padded = pad_sequences(sequence, maxlen=200, padding='post', truncating='post')
        
        # Make prediction
        prediction = model.predict(padded)
        
        # Get human-readable result
        result = "unreliable" if prediction > 0.5 else "reliable"
        confidence = float(prediction[0][0]) if result == "unreliable" else 1 - float(prediction[0][0])
        
        return result, confidence
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None

# Streamlit UI
st.title('Indonesian Fake News Detector')
st.markdown("""
**Enter news text to check its authenticity**
""")

input_text = st.text_area("News Text Input", "", height=200)

if st.button('Check'):
    if len(input_text.strip()) < 10:
        st.warning("Enter longer text (minimum 10 characters)")
    else:
        with st.spinner('Analyzing...'):
            result, confidence = predict_news(input_text)
        
        if result:
            st.subheader("Analysis Results")
            if result == "reliable":
                st.success(f"✅ Reliable News (Confidence: {confidence*100:.1f}%)")
            else:
                st.error(f"❌ Unreliable News (Confidence: {confidence*100:.1f}%)")
            
            st.markdown("""
            **Interpretation of Results:**
            - Confidence level shows how confident the model is with its predictions
            - The analysis results are probabilistic and not a definite diagnosis
            - Always verify through official sources
            """)

st.markdown("""
---
**Disclaimer:**
    - This application uses a machine learning model to predict the authenticity of news
    - Prediction results are probabilistic and not absolute certainty
    - Always verify information through official sources
""")