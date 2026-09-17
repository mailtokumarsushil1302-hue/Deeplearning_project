import os
import json
import h5py
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
import gdown

MODEL_PATH = "brain_tumor_model.h5"
GDRIVE_FILE_ID = "1vFKHK9GPeYPbtNSTXUgCBnd0OmvUxbTO"  # Replace with actual Google Drive ID when ready


def clean_h5_config(file_path):
    """Removes incompatible Keras 3 metadata from the H5 file attributes."""
    try:
        with h5py.File(file_path, "r+") as f:
            if "model_config" in f.attrs:
                config = f.attrs["model_config"]
                if isinstance(config, bytes):
                    config = config.decode("utf-8")

                # Parse JSON model structure
                config_dict = json.loads(config)

                # Recursively clean layer configs
                def clean_node(node):
                    if isinstance(node, dict):
                        node.pop("quantization_config", None)
                        if "config" in node and isinstance(node["config"], dict):
                            node["config"].pop("quantization_config", None)
                            if "batch_shape" in node["config"]:
                                b_shape = node["config"].pop("batch_shape")
                                if b_shape and b_shape[0] is None:
                                    node["config"]["input_shape"] = b_shape[1:]
                        for k, v in node.items():
                            clean_node(v)
                    elif isinstance(node, list):
                        for item in node:
                            clean_node(item)

                clean_node(config_dict)
                f.attrs["model_config"] = json.dumps(config_dict).encode("utf-8")
    except Exception as e:
        st.warning(f"Note: Model cleaning skipped or failed: {e}")


@st.cache_resource
def load_tumor_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

    # Clean metadata before deserialization
    clean_h5_config(MODEL_PATH)

    return tf.keras.models.load_model(MODEL_PATH, compile=False)


st.title("Brain Tumor Detection System")
st.write("Upload an MRI scan to check for brain tumor.")

model = load_tumor_model()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    # Ensure image has 3 color channels (RGB)
    if image.mode != "RGB":
        image = image.convert("RGB")

    img = image.resize((128, 128))  # Verify target size matches model input
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)

    # Define class names (update these to match your model's exact labels)
    class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

    if predictions.shape[-1] > 1:
        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = np.max(predictions[0]) * 100
    else:
        predicted_class = "Tumor Detected" if predictions[0][0] > 0.5 else "No Tumor"
        confidence = (predictions[0][0] if predictions[0][0] > 0.5 else 1 - predictions[0][0]) * 100

    st.success(f"**Prediction:** {predicted_class}")
    st.info(f"**Confidence:** {confidence:.2f}%")