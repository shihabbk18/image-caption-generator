import streamlit as st
from PIL import Image
import numpy as np
import pickle
import os
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(page_title="Image Caption Generator", page_icon="📸", layout="centered")

st.title("📸 Image Caption Generator")
st.markdown("**InceptionV3 + LSTM** • Trained on Flickr8k")

# ====================== Load Models ======================
@st.cache_resource
def load_feature_extractor():
    base_model = InceptionV3(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
    return model

@st.cache_resource
def load_caption_model():
    return tf.keras.models.load_model('models/model.h5')

@st.cache_resource
def load_tokenizer():
    with open('features/tokenizer.pkl', 'rb') as f:
        return pickle.load(f)

feature_extractor = load_feature_extractor()
caption_model = load_caption_model()
tokenizer = load_tokenizer()

max_length = 34  # Change if your training used different value

# ====================== Functions ======================
def extract_features_from_image(image):
    image = image.resize((299, 299))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    features = feature_extractor.predict(image, verbose=0)
    return features

def generate_caption(photo_features):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        
        yhat = caption_model.predict([photo_features, sequence], verbose=0)
        yhat = np.argmax(yhat)
        
        word = None
        for w, idx in tokenizer.word_index.items():
            if idx == yhat:
                word = w
                break
        if word is None:
            break
            
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text.replace('startseq ', '').replace(' endseq', '')

# ====================== Main UI ======================
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with col2:
        if st.button("🚀 Generate Caption", type="primary"):
            with st.spinner("Extracting image features..."):
                features = extract_features_from_image(image)
            
            with st.spinner("Generating caption..."):
                caption = generate_caption(features)
            
            st.success("**Caption:**")
            st.write(f"**{caption}**")
            
            if st.button("Copy Caption"):
                st.code(caption)

# Sidebar
st.sidebar.title("About")
st.sidebar.info(
    "This project uses InceptionV3 for image features and LSTM for caption generation.\n\n"
    "Dataset: Flickr8k"
)
st.sidebar.caption("Made with ❤️ using TensorFlow & Streamlit")