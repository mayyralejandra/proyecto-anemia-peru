import streamlit as st
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# ─── Ruta robusta al modelo (funciona en local Y en Streamlit Cloud) ──────────
BASE_DIR = Path(__file__).parent.parent  # sube un nivel desde app/ hasta la raíz del repo
MODEL_PATH = BASE_DIR / "models" / "modelo_final.pkl"

# ─── Configuración de la página ───────────────────────────────────────────────
st.set_page_config(
    page_title="Diagnóstico de Anemia",
    page_icon="🩸",
    layout="centered"
)

# ─── Cargar modelo ────────────────────────────────────────────────────────────
@st.cache_resource
def cargar_modelo():
    return joblib.load(MODEL_PATH)

artefacto = cargar_modelo()
modelo   = artefacto["modelo"]
le       = artefacto["label_encoder"]
features = artefacto["features"]

# ─── Título y descripción ─────────────────────────────────────────────────────
st.title("🩸 Predictor de Diagnóstico de Anemia")
st.markdown(
    """
    **Proyecto Integrador — Ciencia de Datos**  
    Este sistema predice el diagnóstico de anemia en niños menores de 5 años
    a partir de sus datos clínicos y socioeconómicos, usando un modelo de
    **Random Forest** entrenado con datos del SIEN – Ayacucho, Perú.
    """
)
st.divider()

# ─── Formulario de entrada ────────────────────────────────────────────────────
st.subheader("📋 Ingrese los datos del paciente")

col1, col2 = st.columns(2)

with col1:
    sexo   = st.radio("Sexo", options=["Masculino", "Femenino"], horizontal=True)
    edad   = st.number_input("Edad (meses)", min_value=0, max_value=60, value=24, step=1)
    hbc    = st.number_input("Hemoglobina corregida - Hbc (g/dL)",
                             min_value=3.0, max_value=20.0, value=11.5, step=0.1,
                             help="Hemoglobina ajustada por altitud")
    altura = st.number_input("Altitud del lugar de residencia (msnm)",
                             min_value=0, max_value=5000, value=2750, step=50)

with col2:
    cred       = st.selectbox("¿Tiene CRED al día?",            options=[1, 0], format_func=lambda x: "Sí" if x == 1 else "No")
    suplemento = st.selectbox("¿Recibe suplemento de hierro?",  options=[1, 0], format_func=lambda x: "Sí" if x == 1 else "No")
    juntos     = st.selectbox("¿Pertenece al programa Juntos?", options=[0, 1], format_func=lambda x: "Sí" if x == 1 else "No")
    sis        = st.selectbox("¿Tiene SIS?",                    options=[1, 0], format_func=lambda x: "Sí" if x == 1 else "No")
    qaliwarma  = st.selectbox("¿Pertenece a Qali Warma?",       options=[0, 1], format_func=lambda x: "Sí" if x == 1 else "No")

st.divider()

# ─── Predicción ───────────────────────────────────────────────────────────────
if st.button("🔍 Predecir diagnóstico", use_container_width=True, type="primary"):

    sexo_val = 1 if sexo == "Masculino" else 0

    entrada = pd.DataFrame([[sexo_val, edad, altura, hbc,
                              cred, suplemento, juntos, sis, qaliwarma]],
                           columns=features)

    pred     = modelo.predict(entrada)[0]
    probs    = modelo.predict_proba(entrada)[0]
    etiqueta = le.inverse_transform([pred])[0]

    color_map = {
        "Normal":          "✅",
        "Anemia Leve":     "🟡",
        "Anemia Moderada": "🟠",
        "Anemia Severa":   "🔴"
    }
    icono = color_map.get(etiqueta, "❓")

    st.subheader("📊 Resultado")
    st.metric("Diagnóstico predicho", f"{icono} {etiqueta}")

    st.subheader("📈 Probabilidades por clase")
    prob_df = pd.DataFrame({
        "Diagnóstico": le.classes_,
        "Probabilidad (%)": (probs * 100).round(2)
    }).sort_values("Probabilidad (%)", ascending=False).reset_index(drop=True)

    st.dataframe(prob_df, use_container_width=True, hide_index=True)
    st.bar_chart(prob_df.set_index("Diagnóstico")["Probabilidad (%)"])

    if etiqueta != "Normal":
        st.warning("⚠️ Este resultado es una predicción orientativa. "
                   "Consulte siempre con un profesional de salud para el diagnóstico oficial.")
    else:
        st.success("El paciente no presenta indicios de anemia según el modelo.")

# ─── Pie de página ────────────────────────────────────────────────────────────
st.divider()
st.caption("Modelo: Random Forest | Dataset: SIEN Ayacucho | Proyecto Integrador — Ciencia de Datos")
