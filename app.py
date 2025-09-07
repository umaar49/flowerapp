import numpy as np
import pandas as pd
import tensorflow as tf
import streamlit as st
import cv2
from PIL import Image
import io
import base64

# Set flower background
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://cdn.pixabay.com/photo/2022/05/18/02/16/luminous-7204052_1280.jpg");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    /* Semi-transparent overlay for better readability */
    .stApp::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-color: rgba(255, 255, 255, 0.85);
        z-index: -1;
    }

    /* Style main content area */
    .main .block-container {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-top: 2rem;
        margin-bottom: 2rem;
    }

    /* Style headers */
    h1, h2, h3 {
        color: #4a4a4a;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Your existing app code continues here...
st.title("🌸 Flower Classification")
st.subheader("Detect Different Types of Flowers")

uploaded_file = st.file_uploader("Upload File", type=['png', 'jpg', 'jpeg'])

# ... rest of your existing code ...
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(r"C:\UMAR DATA\Data Sciences\Deep Learning new\projects\model\perfect_model")

MODEL = load_model()  # This will be cached
Classes = ['alpine sea holly', 'anthurium', 'artichoke', 'azalea', 'ball moss', 'balloon flower',
           'barbeton daisy', 'bearded iris', 'bee balm', 'bird of paradise', 'bishop of llandaff',
           'black-eyed susan', 'blackberry lily', 'blanket flower', 'bolero deep blue',
           'bougainvillea', 'bromelia', 'buttercup', 'californian poppy', 'camellia', 'canna lily',
           'canterbury bells', 'cape flower', 'carnation', 'cautleya spicata', 'clematis', "colt's foot",
           'columbine', 'common dandelion', 'corn poppy', 'cyclamen', 'daffodil', 'desert-rose', 'english marigold',
           'fire lily', 'foxglove', 'frangipani', 'fritillary', 'garden phlox', 'gaura', 'gazania',
           'geranium', 'giant white arum lily', 'globe thistle', 'globe-flower', 'grape hyacinth',
           'great masterwort', 'hard-leaved pocket orchid', 'hibiscus', 'hippeastrum', 'japanese anemone',
           'king protea', 'lenten rose', 'lotus lotus', 'love in the mist', 'magnolia', 'mallow',
           'marigold', 'mexican aster', 'mexican petunia', 'monkshood', 'moon orchid', 'morning glory',
           'orange dahlia', 'osteospermum', 'oxeye daisy', 'passion flower', 'pelargonium', 'peruvian lily',
           'petunia', 'pincushion flower', 'pink primrose', 'pink-yellow dahlia', 'poinsettia', 'primula',
           'prince of wales feathers', 'purple coneflower', 'red ginger', 'rose', 'ruby-lipped cattleya',
           'siam tulip', 'silverbush', 'snapdragon', 'spear thistle', 'spring crocus', 'stemless gentian',
           'sunflower', 'sweet pea', 'sweet william', 'sword lily', 'thorn apple', 'tiger lily', 'toad lily',
           'tree mallow', 'tree poppy', 'trumpet creeper', 'wallflower', 'water lily', 'watercress', 'wild pansy',
           'windflower', 'yellow iris']

@st.cache_data
def process_image(uploaded_file):
    # Read the file bytes
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)

    # Reset the file pointer to the beginning for future reads
    uploaded_file.seek(0)

    # Decode the image
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Resize and normalize the image
    resize_image = cv2.resize(image, (224, 224))
    resize_image = np.array(resize_image, dtype='float32')
    normalize_image = resize_image / 255.0

    # Add batch dimension
    input_image = np.expand_dims(normalize_image, axis=0)

    # Make prediction
    predictions = MODEL.predict(input_image)
    predicted_class = Classes[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    return predicted_class, confidence, image


if uploaded_file is not None:
    # Create a copy of the uploaded file to work with
    file_bytes = uploaded_file.read()

    # Display the uploaded image
    image = Image.open(io.BytesIO(file_bytes))
    st.image(image, caption='Uploaded Image',width=448)

    # Create a new file-like object from the bytes for processing
    file_for_processing = io.BytesIO(file_bytes)

    # Process the image and make prediction
    predicted_class, confidence, opencv_image = process_image(file_for_processing)

    # Display the prediction
    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}%")
