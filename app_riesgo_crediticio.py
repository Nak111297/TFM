import streamlit as st
import pickle
import numpy as np
import xgboost as xgb
import os
import requests
import shap
import matplotlib.pyplot as plt

# Configuración de la página
st.set_page_config(page_title="Evaluador de Riesgo Crediticio", page_icon="💼", layout="centered")
st.markdown("""
    <style>
    html, body {
        background-color: #ffffff !important;
        color: #000000 !important;
        font-family: 'Arial', sans-serif;
    }
    [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.6rem 1.2rem;
        font-size: 1rem;
        border: none;
    }
    input, select, textarea {
        background-color: #f9f9f9 !important;
        color: #000000 !important;
        border-radius: 5px;
        border: 1px solid #ccc;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6, .stMarkdown p {
        color: #000000 !important;
    }
    .stTextInput>div>input, .stNumberInput input {
        background-color: #f9f9f9 !important;
        color: #000000 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Cargar modelo
with open("xgb_model_credit_risk.pkl", "rb") as f:
    model = pickle.load(f)

# Título
st.title("💼 Evaluador de Riesgo Crediticio")
st.write("Completa los datos financieros para recibir una evaluación personalizada del riesgo crediticio y una explicación generada por inteligencia artificial.")

# Inputs del usuario
with st.container():
    st.subheader("📄 Información del solicitante")
    col1, col2 = st.columns(2)
    with col1:
        edad = st.number_input("Edad", 18, 100, 30)
        ingresos = st.number_input("Ingresos Anuales ($)", 0.0, 200000.0, 40000.0, step=500.0)
        experiencia = st.number_input("Años de experiencia laboral", 0.0, 50.0, 5.0)
        home_ownership = st.selectbox("Tipo de propiedad", ["OWN", "RENT", "MORTGAGE", "OTHER"])
        genero = st.selectbox("Género", ["Femenino", "Masculino"])
    with col2:
        monto_prestamo = st.number_input("Monto del préstamo ($)", 0.0, 100000.0, 10000.0, step=1000.0)
        nivel_educativo = st.selectbox("Nivel educativo", ["High School", "Associate", "Bachelor", "Master", "Doctorate"])
        loan_intent = st.selectbox("Finalidad del préstamo", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
        historial_crediticio = st.number_input("Años de historial crediticio", 0, 50, 5)

# Mapeo de variables
map_educacion = {'High School': 1, 'Associate': 2, 'Bachelor': 3, 'Master': 4, 'Doctorate': 5}
genero_binario = 0 if genero == "Femenino" else 1
is_young = 1 if edad < 25 else 0

# Transformaciones
log_income = np.log1p(ingresos)
loan_to_income_ratio = monto_prestamo / (ingresos + 1)
income_per_year_of_exp = ingresos / (experiencia if experiencia > 0 else 0.1)

# Dummies para home_ownership y loan_intent (sin OWN porque fue drop_first)
home_ownership_cols = ['person_home_ownership_MORTGAGE', 'person_home_ownership_OTHER', 'person_home_ownership_RENT']
loan_intent_cols = ['loan_intent_EDUCATION', 'loan_intent_HOMEIMPROVEMENT', 'loan_intent_MEDICAL', 'loan_intent_PERSONAL', 'loan_intent_VENTURE']

home_ownership_map = {col: 0 for col in home_ownership_cols}
home_col_key = f"person_home_ownership_{home_ownership}"
if home_col_key in home_ownership_map:
    home_ownership_map[home_col_key] = 1

loan_intent_map = {col: 0 for col in loan_intent_cols}
intent_col_key = f"loan_intent_{loan_intent}"
if intent_col_key in loan_intent_map:
    loan_intent_map[intent_col_key] = 1

# Construcción del input
X = np.array([[
    edad, genero_binario, map_educacion[nivel_educativo], ingresos, experiencia, monto_prestamo,
    historial_crediticio, loan_to_income_ratio, income_per_year_of_exp, log_income, is_young,
    home_ownership_map['person_home_ownership_MORTGAGE'],
    home_ownership_map['person_home_ownership_OTHER'],
    home_ownership_map['person_home_ownership_RENT'],
    loan_intent_map['loan_intent_EDUCATION'],
    loan_intent_map['loan_intent_HOMEIMPROVEMENT'],
    loan_intent_map['loan_intent_MEDICAL'],
    loan_intent_map['loan_intent_PERSONAL'],
    loan_intent_map['loan_intent_VENTURE']
]])

# Predicción
st.markdown("---")
if st.button("🔍 Evaluar Riesgo"):
    try:
        proba = model.predict_proba(X)[0][1]
        riesgo = "Alto" if proba > 0.5 else "Bajo"

        st.success(f"Riesgo crediticio estimado: **{riesgo}**")
        st.markdown(f"### Probabilidad estimada de impago: `{proba:.2%}`")

        # Explicación SHAP local (waterfall plot)
        feature_names = [
            "edad", "genero", "educacion", "ingresos", "experiencia", "monto_prestamo",
            "historial_crediticio", "loan_to_income_ratio", "income_per_year_of_exp", "log_income", "is_young",
            *home_ownership_cols, *loan_intent_cols
        ]
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        st.markdown("### 📊 Explicabilidad del modelo (SHAP Waterfall)")
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[0], feature_names=feature_names, show=False)
        st.pyplot(fig)

        # Hugging Face
        hf_token = os.getenv("HF_TOKEN") or st.secrets.get("HF_TOKEN")
        headers = {"Authorization": f"Bearer {hf_token}"}
        api_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"

        prompt = f"""
Explica por qué una persona con los siguientes datos tiene un riesgo crediticio {riesgo}:

- Edad: {edad}
- Género: {genero}
- Ingresos mensuales: ${ingresos}
- Monto del préstamo: ${monto_prestamo}
- Años de experiencia: {experiencia}
- Nivel educativo: {nivel_educativo}
- Tipo de propiedad: {home_ownership}
- Finalidad del préstamo: {loan_intent}
- Años de historial crediticio: {historial_crediticio}
- Probabilidad de impago estimada: {proba:.2%}

Resume en un párrafo profesional y claro.
"""
        response = requests.post(api_url, headers=headers, json={"inputs": prompt})
        response.raise_for_status()
        output = response.json()

        explicacion = output[0]["generated_text"] if isinstance(output, list) else output.get("generated_text", "Sin respuesta")

        if prompt.strip() in explicacion:
            explicacion = explicacion.replace(prompt.strip(), "").strip()

        st.markdown("### 💬 Explicación generada por IA:")
        st.info(explicacion)

    except Exception as e:
        st.error(f"No se pudo generar la explicación: {str(e)}")
