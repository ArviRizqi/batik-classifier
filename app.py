from huggingface_hub import hf_hub_download
import streamlit as st
import torch
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from PIL import Image
import os

st.set_page_config(page_title="Klasifikasi Batik", layout="centered")
st.title("🧵 Klasifikasi Batik Indonesia")

# Setup
REPO_ID = "Artz-03/batik-model"  # Ganti dengan repo kamu
FILENAME = "batik_model.pth"

# Download model dari Hugging Face Hub (hanya pertama kali)
MODEL_PATH = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)

# Load model
@st.cache_resource
def load_model():
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.fc = torch.nn.Linear(model.fc.in_features, 20)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    transform = weights.transforms()
    return model, transform

model, transform = load_model()

CLASS_NAMES = [
    'batik-bali', 'batik-lasem', 'batik-betawi', 'batik-megamendung',
    'batik-pekalongan', 'batik-parang', 'batik-kawung', 'batik-ciamis',
    'batik-priangan', 'batik-gentongan', 'batik-keraton', 'batik-sidomukti',
    'batik-ceplok', 'batik-celup', 'batik-tambal', 'batik-cendrawasih',
    'batik-garutan', 'batik-sidoluhur', 'batik-sekar', 'batik-sogan'
]

uploaded_file = st.file_uploader("Unggah gambar batik", type=["jpg", "png", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diunggah", use_container_width=True)
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        prediction = CLASS_NAMES[predicted.item()]
        st.success(f"✅ Prediksi: **{prediction}**")
