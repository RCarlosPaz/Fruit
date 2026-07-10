import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import io
import os
import gdown
import subprocess
import threading
import time
from pyngrok import ngrok

# --- Configuración del dispositivo (CPU para Streamlit Cloud si no hay GPU) ---
device = torch.device('cpu') # En Streamlit Cloud, es más probable que uses CPU

# --- ID del archivo de Google Drive para el modelo --- ¡IMPORTANTE! MODIFICA ESTO
# Reemplaza 'YOUR_GOOGLE_DRIVE_FILE_ID_HERE' con el ID real de tu modelo
GOOGLE_DRIVE_FILE_ID = '1Zvz1qtI0nPyMZSpL_Hywc6bJe7RaqRDr'
MODEL_PATH = 'resnet18_multiclase.pth'

# --- Definición de la arquitectura del modelo (DEBE SER LA MISMA QUE SE ENTRENÓ) ---
@st.cache_resource(show_spinner="Cargando modelo y pesos...") # Cachea el modelo para que no se cargue en cada ejecución
def load_model():
    # Descargar el modelo de Google Drive si no existe localmente
    if not os.path.exists(MODEL_PATH):
        st.info(f"Descargando el modelo '{MODEL_PATH}' de Google Drive. Esto puede tomar un momento...")
        try:
            gdown.download(id=GOOGLE_DRIVE_FILE_ID, output=MODEL_PATH, quiet=False)
            st.success("¡Modelo descargado exitosamente!")
        except Exception as e:
            st.error(f"Error al descargar el modelo de Google Drive: {e}. Asegúrate de que el ID sea correcto y el archivo esté público.")
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
        st.error(f"Error: El archivo del modelo '{MODEL_PATH}' no se encontró después de la descarga o la ruta es incorrecta.")
        st.stop()

    model.eval() # Poner el modelo en modo evaluación
    model = model.to(device)
    return model


# --- Transformaciones para una sola imagen de inferencia ---
transforme_inferencia = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Mapeo de clases (DEBE COINCIDIR CON EL ORDEN DE ENTRENAMIENTO) ---
clases = ['Apple', 'Banana', 'Orange', 'Pear'] # Asegúrate de que esto coincida con train_dataset.classes

# --- Código de la aplicación Streamlit --- 
# --- Interfaz de Streamlit con diseño personalizado ---
st.set_page_config(
    page_title="Clasificador Multiclase de Frutas",
    page_icon="🍏",
    layout="wide", # Usa un diseño amplio para más espacio
    initial_sidebar_state="expanded"
)

# --- Inyección de CSS personalizado para un tema más armonioso y robusto ---
st.markdown("""
<style>
    :root {
        --primary-fruit-color: #6a994e;
        --background-light: #f0f2f6;
        --secondary-bg-pastel: #d4e09b;
        --text-dark: #2b2b2b;
        --font-family-primary: 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
    }

    body { font-family: var(--font-family-primary); color: var(--text-dark); }

    h1 {
        color: var(--primary-fruit-color);
        text-align: center;
        padding-bottom: 25px;
        border-bottom: 3px solid var(--primary-fruit-color);
        margin-bottom: 30px;
        font-size: 2.5em;
    }
    h2 { color: var(--primary-fruit-color); font-size: 2em; margin-top: 25px; margin-bottom: 15px;}
    h3 { color: var(--text-dark); font-size: 1.5em; margin-top: 20px; margin-bottom: 10px;}

    .main .block-container {
        padding-top: 30px;
        padding-right: 30px;
        padding-left: 30px;
        padding-bottom: 30px;
        background-color: white;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }

    .css-1d391kg {
        background-color: var(--secondary-bg-pastel);
        color: var(--text-dark);
        border-right: 2px solid var(--primary-fruit-color);
        padding-top: 20px;
        padding-left: 15px;
        padding-right: 15px;
    }
    .css-pkz34x { color: var(--text-dark); }

    .stButton>button {
        background-color: var(--primary-fruit-color);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 12px 25px;
        font-size: 1.1em;
        font-weight: bold;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #5b8440;
        color: white;
    }

    .stSuccess {
        background-color: #e6ffe6;
        color: #006400;
        border-radius: 8px;
        padding: 15px;
        margin-top: 15px;
        border-left: 6px solid var(--primary-fruit-color);
        font-size: 1.1em;
        font-weight: 500;
    }
    .stInfo {
        background-color: #e0f2f7;
        color: #007bff;
        border-radius: 8px;
        padding: 15px;
        margin-top: 15px;
        border-left: 6px solid #007bff;
        font-size: 1em;
    }
    .stError {
        background-color: #ffe6e6;
        color: #dc3545;
        border-radius: 8px;
        padding: 15px;
        margin-top: 15px;
        border-left: 6px solid #dc3545;
        font-size: 1.1em;
        font-weight: 500;
    }

    .stMarkdown, .stText, .stLabel {
        color: var(--text-dark);
        font-family: var(--font-family-primary);
        line-height: 1.6;
    }

    .css-fg4pbf {
        background-color: var(--background-light);
        border: 2px dashed var(--primary-fruit-color);
        color: var(--primary-fruit-color);
        padding: 30px;
        border-radius: 12px;
        text-align: center;
        font-size: 1.2em;
        margin-top: 20px;
    }
    .css-fg4pbf:hover {
        background-color: #c0cc94;
    }

    .stCaption {
        font-size: 0.9em;
        color: #666;
        font-style: italic;
        margin-top: 20px;
    }
    
    .stSuccess p {
        font-size: 1.2em;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- Cabecera principal --- 
st.title("🍎🍏🍊 Clasificador de Frutas Multiclase 🍐🍋🍓")
st.markdown("### _Una herramienta inteligente para identificar y clasificar diferentes tipos de frutas._")
st.write("\n") # Espacio para separación

# --- Barra lateral de Información --- 
st.sidebar.header("Acerca de esta Aplicación")
st.sidebar.markdown(
    "Esta aplicación utiliza un modelo de **aprendizaje profundo (CNN ResNet18)** "
    "para clasificar imágenes de frutas en cuatro categorías: "
    "**Manzana (Apple), Plátano (Banana), Naranja (Orange) y Pera (Pear)**."
)
st.sidebar.markdown(
    "\n\n**Guía de Uso:**\n"
    "1. Sube una imagen clara de una fruta usando la sección de 'Carga de Imagen'.\n"
    "2. La aplicación procesará la imagen y mostrará la predicción del modelo con su nivel de confianza.\n"
)
st.sidebar.info(
    "Desarrollado con ❤️ y PyTorch. "
    "El código fuente y más información están disponibles en "
    "[GitHub](https://github.com/tu_usuario/tu_repo_frutas). ¡No olvides reemplazar el enlace!"
)

# --- Sección Principal: Carga y Predicción ---
st.header("1. Sube una Imagen de Fruta")
with st.container(): # Contenedor para la sección de carga
    uploaded_file = st.file_uploader(
        "Arrastra y suelta tu imagen aquí o haz clic para buscar en tu dispositivo", 
        type=["jpg", "jpeg", "png"],
        help="Sube una imagen de una sola fruta clara para una mejor predicción."
    )

if uploaded_file is not None:
    # Leer la imagen subida
    image_data = uploaded_file.getvalue()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")

    st.markdown("--- ") # Separador visual
    st.header("2. Análisis y Resultados")
    
    # Usar columnas para una maquetación más limpia y equilibrada
    col_img, col_res = st.columns([1, 1]) 

    with col_img:
        st.subheader("Imagen Seleccionada")
        st.image(image, caption='Imagen de tu fruta', use_column_width=True)
        st.markdown("\n") # Espacio

    # Preprocesar la imagen
    input_tensor = transforme_inferencia(image)
    input_batch = input_tensor.unsqueeze(0) # Añadir una dimensión de batch
    input_batch = input_batch.to(device)

    # Cargar el modelo aquí, ya que Streamlit lo maneja con @st.cache_resource
    model = load_model()

    # Realizar la predicción
    with torch.no_grad():
        output = model(input_batch)
        probabilities = torch.softmax(output, dim=1)
        _, predicted_idx = torch.max(probabilities, 1)

    # Obtener el nombre de la clase y la confianza
    predicted_class = clases[predicted_idx.item()]
    confidence = probabilities[0][predicted_idx.item()].item() * 100

    with col_res:
        st.subheader("Resultado de la Predicción")
        st.success(f"✨ Predicción: ¡Es una **{predicted_class}**!")
        st.write(f"📊 Nivel de Confianza: **{confidence:.2f}%**")
        st.balloons() # Pequeña celebración

        st.markdown("\n--- ") # Separador visual
        st.subheader("Detalle de Probabilidades:")
        
        # Mostrar todas las probabilidades en una tabla o lista más legible
        prob_data_list = []
        for i, class_name in enumerate(clases):
            prob_data_list.append({"Fruta": class_name, "Probabilidad": f"{probabilities[0][i].item()*100:.2f}%"})
        st.table(prob_data_list)

    st.markdown("--- ")
    st.caption("💡 *Nota Importante: La precisión de este modelo puede variar. "
               "Utiliza esta herramienta como un apoyo para la clasificación de frutas y no "
               "como una fuente definitiva para decisiones críticas.*")

# --- Ejecución del Streamlit App en Colab con Ngrok ---

# Crear el directorio si no existe (para guardar app.py)
output_dir = "/mount/src/fruit"
os.makedirs(output_dir, exist_ok=True)
app_file_path = os.path.join(output_dir, "app.py")

# Guardar el contenido de la aplicación Streamlit en un archivo
# Esto es necesario para que `streamlit run` pueda ejecutarlo.
app_code = """
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import io
import os
import gdown

# --- Configuración del dispositivo (CPU para Streamlit Cloud si no hay GPU) ---
device = torch.device('cpu')

# --- ID del archivo de Google Drive para el modelo --- ¡IMPORTANTE! MODIFICA ESTO
GOOGLE_DRIVE_FILE_ID = '1Zvz1qtI0nPyMZSpL_Hywc6bJe7RaqRDr'
MODEL_PATH = 'resnet18_multiclase.pth'

# --- Definición de la arquitectura del modelo (DEBE SER LA MISMA QUE SE ENTRENÓ) ---
@st.cache_resource(show_spinner="Cargando modelo y pesos...")
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.info(f"Descargando el modelo '{MODEL_PATH}' de Google Drive. Esto puede tomar un momento...")
        try:
            gdown.download(id=GOOGLE_DRIVE_FILE_ID, output=MODEL_PATH, quiet=False)
            st.success("¡Modelo descargado exitosamente!")
        except Exception as e:
            st.error(f"Error al descargar el modelo de Google Drive: {e}. Asegúrate de que el ID sea correcto y el archivo esté público.")
            st.stop()
    model = models.resnet18(weights=None)
    for param in model.parameters():
        param.requires_grad = False
    num_caracteristicas = model.fc.in_features
    model.fc = nn.Linear(num_caracteristicas, 4)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except FileNotFoundError:
        st.error(f"Error: El archivo del modelo '{MODEL_PATH}' no se encontró después de la descarga o la ruta es incorrecta.")
        st.stop()
    model.eval()
    model = model.to(device)
    return model

# --- Transformaciones para una sola imagen de inferencia ---
transforme_inferencia = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Mapeo de clases (DEBE COINCIDIR CON EL ORDEN DE ENTRENAMIENTO) ---
clases = ['Apple', 'Banana', 'Orange', 'Pear']

# --- Configuración y diseño de la interfaz de Streamlit ---
st.set_page_config(
    page_title="Clasificador Multiclase de Frutas",
    page_icon="🍏",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    :root {
        --primary-fruit-color: #6a994e;
        --background-light: #f0f2f6;
        --secondary-bg-pastel: #d4e09b;
        --text-dark: #2b2b2b;
        --font-family-primary: 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
    }

    body { font-family: var(--font-family-primary); color: var(--text-dark); }

    h1 {
        color: var(--primary-fruit-color);
        text-align: center;
        padding-bottom: 25px;
        border-bottom: 3px solid var(--primary-fruit-color);
        margin-bottom: 30px;
        font-size: 2.5em;
    }
    h2 { color: var(--primary-fruit-color); font-size: 2em; margin-top: 25px; margin-bottom: 15px;}
    h3 { color: var(--text-dark); font-size: 1.5em; margin-top: 20px; margin-bottom: 10px;}

    .main .block-container {
        padding-top: 30px;
        padding-right: 30px;
        padding-left: 30px;
        padding-bottom: 30px;
        background-color: white;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }

    .css-1d391kg {
        background-color: var(--secondary-bg-pastel);
        color: var(--text-dark);
        border-right: 2px solid var(--primary-fruit-color);
        padding-top: 20px;
        padding-left: 15px;
        padding-right: 15px;
    }
    .css-pkz34x { color: var(--text-dark); }

    .stButton>button {
        background-color: var(--primary-fruit-color);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 12px 25px;
        font-size: 1.1em;
        font-weight: bold;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #5b8440;
        color: white;
    }

    .stSuccess {
        background-color: #e6ffe6;
        color: #006400;
        border-radius: 8px;
        padding: 15px;
        margin-top: 15px;
        border-left: 6px solid var(--primary-fruit-color);
        font-size: 1.1em;
        font-weight: 500;
    }
    .stInfo {
        background-color: #e0f2f7;
        color: #007bff;
        border-radius: 8px;
        padding: 15px;
        margin-top: 15px;
        border-left: 6px solid #007bff;
        font-size: 1em;
    }
    .stError {
        background-color: #ffe6e6;
        color: #dc3545;
        border-radius: 8px;
        padding: 15px;
        margin-top: 15px;
        border-left: 6px solid #dc3545;
        font-size: 1.1em;
        font-weight: 500;
    }

    .stMarkdown, .stText, .stLabel {
        color: var(--text-dark);
        font-family: var(--font-family-primary);
        line-height: 1.6;
    }

    .css-fg4pbf {
        background-color: var(--background-light);
        border: 2px dashed var(--primary-fruit-color);
        color: var(--primary-fruit-color);
        padding: 30px;
        border-radius: 12px;
        text-align: center;
        font-size: 1.2em;
        margin-top: 20px;
    }
    .css-fg4pbf:hover {
        background-color: #c0cc94;
    }

    .stCaption {
        font-size: 0.9em;
        color: #666;
        font-style: italic;
        margin-top: 20px;
    }
    
    .stSuccess p {
        font-size: 1.2em;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- Cabecera principal ---
st.title("🍎🍏🍊 Clasificador de Frutas Multiclase 🍐🍋🍓")
st.markdown("### _Una herramienta inteligente para identificar y clasificar diferentes tipos de frutas._")
st.write("\n")

# --- Barra lateral de Información ---
st.sidebar.header("Acerca de esta Aplicación")
st.sidebar.markdown(
    "Esta aplicación utiliza un modelo de **aprendizaje profundo (CNN ResNet18)** "
    "para clasificar imágenes de frutas en cuatro categorías: "
    "**Manzana (Apple), Plátano (Banana), Naranja (Orange) y Pera (Pear)**."
)
st.sidebar.markdown(
    "\n\n**Guía de Uso:**\n"
    "1. Sube una imagen clara de una fruta usando la sección de 'Carga de Imagen'.\n"
    "2. La aplicación procesará la imagen y mostrará la predicción del modelo con su nivel de confianza.\n"
)
st.sidebar.info(
    "Desarrollado con ❤️ y PyTorch. "
    "El código fuente y más información están disponibles en "
    "[GitHub](https://github.com/tu_usuario/tu_repo_frutas). ¡No olvides reemplazar el enlace!"
)

# --- Sección Principal: Carga y Predicción ---
st.header("1. Sube una Imagen de Fruta")
with st.container():
    uploaded_file = st.file_uploader(
        "Arrastra y suelta tu imagen aquí o haz clic para buscar en tu dispositivo", 
        type=["jpg", "jpeg", "png"],
        help="Sube una imagen de una sola fruta clara para una mejor predicción."
    )

if uploaded_file is not None:
    image_data = uploaded_file.getvalue()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")

    st.markdown("--- ")
    st.header("2. Análisis y Resultados")
    
    col_img, col_res = st.columns([1, 1]) 

    with col_img:
        st.subheader("Imagen Seleccionada")
        st.image(image, caption='Imagen de tu fruta', use_column_width=True)
        st.markdown("\n")

    input_tensor = transforme_inferencia(image)
    input_batch = input_tensor.unsqueeze(0)
    input_batch = input_batch.to(device)

    model = load_model()

    with torch.no_grad():
        output = model(input_batch)
        probabilities = torch.softmax(output, dim=1)
        _, predicted_idx = torch.max(probabilities, 1)

    predicted_class = clases[predicted_idx.item()]
    confidence = probabilities[0][predicted_idx.item()].item() * 100

    with col_res:
        st.subheader("Resultado de la Predicción")
        st.success(f"✨ Predicción: ¡Es una **{predicted_class}**!")
        st.write(f"📊 Nivel de Confianza: **{confidence:.2f}%**")
        st.balloons()

        st.markdown("\n--- ")
        st.subheader("Detalle de Probabilidades:")
        
        prob_data_list = []
        for i, class_name in enumerate(clases):
            prob_data_list.append({"Fruta": class_name, "Probabilidad": f"{probabilities[0][i].item()*100:.2f}%"})
        st.table(prob_data_list)

    st.markdown("--- ")
    st.caption("💡 *Nota Importante: La precisión de este modelo puede variar. "
               "Utiliza esta herramienta como un apoyo para la clasificación de frutas y no "
               "como una fuente definitiva para decisiones críticas.*")
"""

# Guardar el código de la aplicación Streamlit en un archivo temporal.
# Esto es necesario para que `streamlit run` pueda ejecutarlo.
output_dir = "/mount/src/fruit"
os.makedirs(output_dir, exist_ok=True)
app_file_path = os.path.join(output_dir, "app.py")

with open(app_file_path, "w") as f:
    f.write(app_code)

print(f"Archivo '{app_file_path}' creado exitosamente.")

# --- Configurar y ejecutar Ngrok y Streamlit ---
# Se recomienda obtener un token de autenticación de ngrok desde su sitio web (ngrok.com)
# para evitar limitaciones y usar características avanzadas.
# ngrok.set_auth_token("YOUR_NGROK_AUTH_TOKEN") # Reemplaza con tu token si lo tienes

public_url = ngrok.connect(port=8501)
print(f"\nTu aplicación Streamlit está accesible en: {public_url}\n")

def run_streamlit():
    time.sleep(2) # Pequeño retardo para asegurar que el puerto esté listo
    env = os.environ.copy()
    env["NPM_CONFIG_UPDATE_NOTIFIER"] = "false" # Suprimir notificaciones de NPM
    # Ejecuta Streamlit usando el archivo guardado
    subprocess.run(["python", "-m", "streamlit", "run", app_file_path, "--server.port", "8501", "--server.headless", "true"], env=env)

streamlit_thread = threading.Thread(target=run_streamlit)
streamlit_thread.start()

print("Ejecutando la aplicación Streamlit en segundo plano...")
print("Para detener la aplicación, simplemente detén la ejecución de esta celda.")
