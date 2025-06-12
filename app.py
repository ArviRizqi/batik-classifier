import streamlit as st
import torch
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from PIL import Image
import os
import gdown

st.set_page_config(page_title="Klasifikasi Batik", layout="centered")
st.title("🧵 Klasifikasi Batik Indonesia")
st.write("Unggah gambar batik dan lihat jenisnya berdasarkan model klasifikasi PyTorch.")

# Konstanta
MODEL_ID = "1TcwfUAOUvVCYXwfvKHc9F1UvEEMuaxw2"
MODEL_PATH = "batik_model.pth"

# Download model jika belum ada
if not os.path.exists(MODEL_PATH):
    with st.spinner("Mengunduh model..."):
        gdown.download(id=MODEL_ID, output=MODEL_PATH, quiet=False)

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

# Label
CLASS_NAMES = [
    'batik-bali', 'batik-lasem', 'batik-betawi', 'batik-megamendung',
    'batik-pekalongan', 'batik-parang', 'batik-kawung', 'batik-ciamis',
    'batik-priangan', 'batik-gentongan', 'batik-keraton', 'batik-sidomukti',
    'batik-ceplok', 'batik-celup', 'batik-tambal', 'batik-cendrawasih',
    'batik-garutan', 'batik-sidoluhur', 'batik-sekar', 'batik-sogan'
]

# Upload
uploaded_file = st.file_uploader("Unggah gambar batik", type=["jpg", "png", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diunggah", use_container_width=True)

    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        class_idx = predicted.item()
        prediction = CLASS_NAMES[class_idx]
        st.success(f"✅ Prediksi: **{prediction}**")
