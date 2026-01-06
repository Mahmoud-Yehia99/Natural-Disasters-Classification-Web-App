import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import plotly.graph_objects as go

# ------------------- MUST BE FIRST --------------------
st.set_page_config(page_title="🌪️ Natural Disasters Classifier", page_icon="🌍", layout="centered")

# ------------------- Load EfficientNet Model -------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('results/EfficientNet.keras')
    return model

model = load_model()

# ------------------- Preprocess Function -------------------
def preprocess_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((224, 224))
    img_array = np.array(image)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ------------------- Custom CSS Styling -------------------
st.markdown("""
    <style>
    .main { background-color: #f0f9ff; padding: 20px; border-radius: 15px; }
    .stApp > header { background-color: #00acc1; color: white; text-align: center; }
    .stButton > button {
        background-color: #00acc1;
        color: white;
        border-radius: 10px;
        padding: 10px;
        font-weight: bold;
    }
    img {
        border-radius: 15px;
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.2);
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------- Title & Upload UI -------------------
st.title("Natural Disasters Classification Web App 🌍")
st.markdown("Upload an image to classify it as **Cyclone**, **Earthquake**, **Flood**, or **Wildfire**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# ------------------- Prediction Logic -------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)

    processed = preprocess_image(image)
    prediction = model.predict(processed)[0]
    class_names = ['Cyclone', 'Earthquake', 'Flood', 'Wildfire']

    predicted_class = class_names[np.argmax(prediction)]
    confidence = prediction[np.argmax(prediction)] * 100

    st.success(f"**Prediction:** {predicted_class}")
    st.info(f"**Confidence:** {confidence:.2f}%")

    # ------------------- Plotly Result Chart -------------------
    fig = go.Figure(data=[
        go.Bar(
            x=class_names,
            y=[round(prob * 100, 2) for prob in prediction],
            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        )
    ])
    fig.update_layout(
        title='Prediction Confidence (%)',
        xaxis_title='Disaster Type',
        yaxis_title='Confidence (%)',
        yaxis=dict(range=[0, 100])
    )
    st.plotly_chart(fig)

# ------------------- Sample Images Section -------------------
st.markdown("### 📸 Sample Images")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.image("samples/cyclone.jpg", caption="Cyclone", use_container_width=True)
with col2:
    st.image("samples/earthquake.jpg", caption="Earthquake", use_container_width=True)
with col3:
    st.image("samples/flood.jpg", caption="Flood", use_container_width=True)
with col4:
    st.image("samples/wildfire.jpg", caption="Wildfire", use_container_width=True)

# ------------------- Sidebar Info -------------------
st.sidebar.header("ℹ️ App Info")
st.sidebar.write("""
This web app uses an **EfficientNet** model to classify natural disasters.
It can classify 4 types:
- 🌪 Cyclone
- 🌍 Earthquake
- 🌊 Flood
- 🔥 Wildfire
""")

st.sidebar.header("📢 Tips")
st.sidebar.write("""
- Upload a clear image showing the disaster scene.
- Predictions are AI-based — verify results if used for decision-making.
""")

# ------------------- Sidebar Developer Credit -------------------
st.sidebar.markdown("---")
st.sidebar.markdown("👨‍💻 **Developed by:** Mahmoud Yehia Emam")

# ------------------- Bottom Tip -------------------
st.markdown("***")
st.markdown("🧰 **Preparedness Tip:** Keep emergency supplies and know your local evacuation plans.")
