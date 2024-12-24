import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from scipy.spatial.distance import cosine
import numpy as np

# Define a simple Siamese Network
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 256)  # Output feature vector size 256

    def forward(self, x):
        return self.resnet(x)

# Load the model and set to evaluation mode
model = SiameseNetwork()
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to get image embeddings
def get_image_embedding(image, model, transform):
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        features = model(image)
    return features

# Function to compare two images and return similarity
def compare_images(image1, image2, model, transform):
    embedding1 = get_image_embedding(image1, model, transform).squeeze().cpu().numpy()
    embedding2 = get_image_embedding(image2, model, transform).squeeze().cpu().numpy()
    
    # Calculate cosine similarity
    similarity = 1 - cosine(embedding1, embedding2)
    return similarity

# Heuristic fucntion to check if the image is likely a signature
def is_signature_image(image):
    # convert image to grayscale and check if it's mostly monochrome (low color variance)
    grayscale_image = image.convert("L")
    historgram = grayscale_image.histogram()
    variance = np.var(historgram)

    # Threshold for checking if an image is likely to be a signature
    # Signature often have low variance in color distribution
    if variance < 2000: # you can tweak this threshold based on testing
        return True
    return False 

# Streamlit UI
st.set_page_config(page_title="Comparar Firmas Intermoda", layout="centered", initial_sidebar_state="expanded")

# App Title
st.title("🖋️ Signcheck Powered By AI/ML")
st.write("Sube ambas firmas para verificar que sean la misma.")

# Image file uploaders
col1, col2 = st.columns(2)
with col1:
    st.subheader("Sube la firma 1")
    image1_file = st.file_uploader("Escoge la primer firma", type=["png", "jpg", "jpeg"])

with col2:
    st.subheader("Sube la firma 2")
    image2_file = st.file_uploader("Escoge la segunda firma", type=["png", "jpg", "jpeg"])

# Placeholder for displaying the result
result_placeholder = st.empty()

if image1_file and image2_file:
    # Load images
    image1 = Image.open(image1_file).convert("RGB")
    image2 = Image.open(image2_file).convert("RGB")

    # # validate both images are likely to be signatures
    # if not is_signature_image(image1):
    #     st.error("The first image does not appear to be a signature. Please upload a valid signature")
    
    # elif not is_signature_image(image2):
    #     st.error("The second image does not appear to be a signature. Please upload a valid signature")
    
    # else:
        
    
    # Display images
    st.subheader("Uploaded Signatures:")
    col1, col2 = st.columns(2)
    col1.image(image1, caption="Firma 1", use_column_width=True)
    col2.image(image2, caption="Firma 2", use_column_width=True)
    
    # Compare signatures on button click
    if st.button("Compara las firmas"):
        with st.spinner("Comparando..."):
            similarity_score = compare_images(image1, image2, model, transform)
        
        # Display similarity score
        result_placeholder.subheader(f"Similitud: {similarity_score:.2f}")
        
        # Interpret similarity
        if similarity_score > 0.90:
            st.success("Las firmas son la misma.")
        else:
            st.error("Las firmas son diferentes.")

st.markdown("---")
st.markdown("Las firmas comparadas no se almacenan de ninguna forma. Solo es una app de comparacion.")

# Add a beautiful sidebar
st.sidebar.title("Acerca de la App")
# st.sidebar.write("""
# This app uses a deep learning model, Siamese Network to compare two signature images. 
# It calculates the cosine similarity between the two signature embeddings and provides a similarity score.
# If the score is equals or above a certain threshold (0.98), the signatures are considered similar.
# You can use this app for signature verification cutting across business lines.
# """)
st.sidebar.write("""
Esta app utiliza un modelo de Deep Learning para comparar 2 firmas.
Calcula la similitud entre 2 firmas embebidas y arroja el score de similitud.
Si el score es igual o mayor a cierto criterio de aceptaci[on (0.90), entonces las firmas son consideradas como
autenticas e iguales. Pero si el score es menor a 0.90 entonces se considera que las firmas no cumplen con los requisitos para ser
autenticas o iguales.
""")

st.sidebar.markdown("### Como funciona")
st.sidebar.write("""
- Sube 2 imagenes de firmas.
- La app extrae caracteristicas de ambas imagenes utilizando un modelo de Deep Learning.
- Muestra un score de similitud basado en los emebebidos de las imegenes.
""")

# Add footer with contact info or credits
st.sidebar.markdown("### Credits")
st.sidebar.write("IM - Sofia Videa.")
