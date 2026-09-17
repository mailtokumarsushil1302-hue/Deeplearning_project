import tf_keras
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


# Load model
@st.cache_resource
def load_tumor_model():
    return tf.keras.models.load_model('brain_tumor_model.h5', compile=False)

model = load_tumor_model()
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

st.title('Brain Tumor Detection System')
st.write('Upload a brain MRI scan image to test the model.')

uploaded_file = st.file_uploader(
    'Choose an MRI Image...', type=['jpg', 'jpeg', 'png']
)

if uploaded_file is not None:
  image = Image.open(uploaded_file).convert('RGB')
  st.image(image, caption='Uploaded MRI Scan', use_column_width=True)

  # Preprocess to match training specs (128x128)
  img = image.resize((128, 128))
  img_array = np.array(img) / 255.0
  img_array = np.expand_dims(img_array, axis=0)

  # Predict class
  predictions = model.predict(img_array)
  predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
  confidence = np.max(predictions[0]) * 100

  st.subheader(f'Result: {predicted_class.upper()}')
  st.write(f'Confidence Score: {confidence:.2f}%')