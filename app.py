import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import io
import os
import gdown # Necesitas instalar 'gdown' en tu entorno de Streamlit

# --- Configuración del dispositivo (CPU para Streamlit Cloud si no hay GPU) ---
device = torch.device('cpu') # En Streamlit Cloud, es más probable que uses CPU

# --- ID del archivo de Google Drive para el modelo --- ¡IMPORTANTE! MODIFICA ESTO
GOOGLE_DRIVE_FILE_ID = '1Zvz1qtI0nPyMZSpL_Hywc6bJe7RaqRDr' # <<< REEMPLAZA CON EL ID DE TU ARCHIVO
MODEL_PATH = 'resnet18_multiclase.pth'

# --- Definición de la arquitectura del modelo (DEBE SER LA MISMA QUE SE ENTRENÓ) ---
@st.cache_resource # Cachea el modelo para que no se cargue en cada ejecución
def load_model():
    # Descargar el modelo de Google Drive si no existe localmente
    if not os.path.exists(MODEL_PATH):
        st.info(f"Descargando el modelo '{MODEL_PATH}' de Google Drive...")
        try:
            gdown.download(id=GOOGLE_DRIVE_FILE_ID, output=MODEL_PATH, quiet=False)
            st.success("Modelo descargado exitosamente.")
        except Exception as e:
            st.error(f"Error al descargar el modelo de Google Drive: {e}")
            st.stop() # Detiene la aplicación si no se puede descargar el modelo

    # Cargar la estructura de ResNet18
    model = models.resnet18(weights=None) # No necesitamos los pesos pre-entrenados de ImageNet aquí

    # Congelar las capas base (aunque no entrenaremos, es buena práctica mantener la misma estructura)
    for param in model.parameters():
        param.requires_grad = False

    # Reemplazar la última capa de clasificación
    num_caracteristicas = model.fc.in_features
    model.fc = nn.Linear(num_caracteristicas, 4) # 4 clases (Apple, Banana, Orange, Pear)

    # Cargar los pesos entrenados
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except FileNotFoundError:
        st.error(f"Error: El archivo del modelo '{MODEL_PATH}' no se encontró después de la descarga.")
        st.stop()

    model.eval() # Poner el modelo en modo evaluación
    model = model.to(device)
    return model

model = load_model()

# --- Transformaciones para una sola imagen de inferencia ---
transforme_inferencia = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Mapeo de clases (DEBE COINCIDIR CON EL ORDEN DE ENTRENAMIENTO) ---
clases = ['Apple', 'Banana', 'Orange', 'Pear'] # Asegúrate de que esto coincida con train_dataset.classes

# --- Interfaz de Streamlit ---
st.set_page_config(page_title="Clasificador Multiclase de Frutas", page_icon="🍏")

st.title("🍎 Clasificador Multiclase de Frutas")
st.markdown("Una aplicación para clasificar imágenes de frutas usando un modelo ResNet18 pre-entrenado.")

# Barra lateral
st.sidebar.header("Acerca de")
st.sidebar.markdown("Esta aplicación utiliza un modelo de redes neuronales convolucionales (CNN) basado en ResNet18 para clasificar imágenes de frutas.")
st.sidebar.markdown("\n\n**Cómo usar:**\n1. Sube una imagen en la sección principal.\n2. La aplicación mostrará la imagen y la predicción del modelo.\n")
st.sidebar.markdown("--- desarrollado con ❤️ y PyTorch")

st.header("Sube tu imagen aquí")
uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Leer la imagen subida
    image_data = uploaded_file.getvalue()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")

    # Crear columnas para mostrar la imagen y la predicción
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Imagen Subida")
        st.image(image, caption='Imagen para analizar', use_column_width=True)

    # Preprocesar la imagen
    input_tensor = transforme_inferencia(image)
    input_batch = input_tensor.unsqueeze(0) # Añadir una dimensión de batch
    input_batch = input_batch.to(device)

    # Realizar la predicción
    with torch.no_grad():
        output = model(input_batch)
        probabilities = torch.softmax(output, dim=1)
        _, predicted_idx = torch.max(probabilities, 1)

    # Obtener el nombre de la clase y la confianza
    predicted_class = clases[predicted_idx.item()]
    confidence = probabilities[0][predicted_idx.item()].item() * 100

    with col2:
        st.subheader("Resultado de la Predicción")
        st.success(f"Predicción: **{predicted_class}**")
        st.write(f"Confianza: {confidence:.2f}%")
        st.balloons() # Mantener la celebración

        st.markdown("--- ")
        st.subheader("Probabilidades Completas:")
        for i, class_name in enumerate(clases):
            st.write(f"**{class_name}**: {probabilities[0][i].item()*100:.2f}%")

    st.markdown("--- ")
    st.caption("*Nota: La precisión del modelo puede variar. Usa esto como una herramienta de apoyo.*")
