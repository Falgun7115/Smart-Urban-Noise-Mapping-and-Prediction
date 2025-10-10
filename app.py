import streamlit as st
import numpy as np
import librosa
import joblib
import pandas as pd

st.title("Smart Urban Noise Classification & Mapping")

# Load model
model = joblib.load('urban_noise_model.pkl')
class_names = ["air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling",
               "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"]

# Upload sound
uploaded_file = st.file_uploader("Upload WAV sound file", type=["wav"])
if uploaded_file:
    audio, sr = librosa.load(uploaded_file, sr=None)
    mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0).reshape(1, -1)
    pred_class = model.predict(mfccs)[0]
    st.write(f"Predicted class: {class_names[pred_class]}")
    st.audio(uploaded_file)
else:
    st.info("Please upload a WAV file.")

# Sample mapping visualization
st.markdown("### Sample Noise Locations")
sample_locations = pd.DataFrame({
    "lat": [40.7128, 40.7138, 40.7148],
    "lon": [-74.0060, -74.0050, -74.0070],
    "label": ["car_horn", "siren", "jackhammer"]
})
st.map(sample_locations)
