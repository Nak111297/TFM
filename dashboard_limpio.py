import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns

# Configuración de página
st.set_page_config(page_title="TFM - Evaluación de Riesgo Crediticio", layout="wide")
st.title("📘 TFM - Evaluación de Riesgo Crediticio con Machine Learning")

# Cargar dataset
@st.cache_data
def cargar_datos():
    return pd.read_csv("loan_data.csv")

df = cargar_datos()

# Sidebar - Filtros
st.sidebar.header("🎛️ Filtros del análisis")
if 'person_age' in df.columns:
    edad = st.sidebar.slider("Edad", int(df['person_age'].min()), int(df['person_age'].max()), (int(df['person_age'].min()), 60))
if 'person_income' in df.columns:
    ingreso = st.sidebar.slider("Ingreso", int(df['person_income'].min()), int(df['person_income'].max()), (10000, 60000))
if 'loan_amnt' in df.columns:
    monto = st.sidebar.slider("Monto del Préstamo", int(df['loan_amnt'].min()), int(df['loan_amnt'].max()), (1000, 25000))
if 'credit_score' in df.columns:
    score = st.sidebar.slider("Credit Score", int(df['credit_score'].min()), int(df['credit_score'].max()), (300, 850))
if 'loan_int_rate' in df.columns:
    int_rate = st.sidebar.slider("Tasa de Interés", float(df['loan_int_rate'].min()), float(df['loan_int_rate'].max()), (5.0, 20.0))

# Aplicar filtros si están disponibles
columnas_requeridas = ['person_age', 'person_income', 'loan_amnt', 'credit_score', 'loan_int_rate']
if all(col in df.columns for col in columnas_requeridas):
    df = df[
        df['person_age'].between(*edad) &
        df['person_income'].between(*ingreso) &
        df['loan_amnt'].between(*monto) &
        df['credit_score'].between(*score) &
        df['loan_int_rate'].between(*int_rate)
    ]

# Tabs de contenido
tabs = st.tabs(["📊 Exploración", "📈 Correlaciones", "🤖 Evaluación del Modelo", "📌 Conclusiones"])

# Exploración Visual
with tabs[0]:
    st.header("📊 Exploración Visual de Variables")
    col1, col2 = st.columns(2)
    if 'loan_amnt' in df.columns:
        with col1:
            fig1 = px.histogram(df, x='loan_amnt', nbins=30, title="Distribución del Monto del Préstamo")
            st.plotly_chart(fig1, use_container_width=True)
    if 'loan_int_rate' in df.columns:
        with col2:
            fig2 = px.histogram(df, x='loan_int_rate', nbins=30, title="Distribución de la Tasa de Interés")
            st.plotly_chart(fig2, use_container_width=True)

    if 'person_income' in df.columns and 'credit_score' in df.columns:
        st.subheader("📌 Ingreso vs. Credit Score")
        fig3 = px.scatter(df, x='person_income', y='credit_score', color='loan_status' if 'loan_status' in df.columns else None)
        st.plotly_chart(fig3, use_container_width=True)

# Correlaciones
with tabs[1]:
    st.header("🔗 Matriz de Correlación")
    variables_numericas = df.select_dtypes(include='number')
    if not variables_numericas.empty:
        fig_corr, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(variables_numericas.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig_corr)

# Evaluación del modelo si existe 'score_model'
with tabs[2]:
    st.header("🤖 Evaluación del Modelo de Machine Learning")
    if 'score_model' in df.columns and 'loan_status' in df.columns:
        y_true = df['loan_status']
        y_score = df['score_model']
        y_pred = (y_score >= 0.5).astype(int)

        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        st.subheader("Curva ROC")
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        ax_roc.plot([0, 1], [0, 1], 'k--')
        ax_roc.set_xlabel("FPR")
        ax_roc.set_ylabel("TPR")
        ax_roc.set_title("Curva ROC")
        ax_roc.legend()
        st.pyplot(fig_roc)

        st.subheader("Matriz de Confusión")
        cm = confusion_matrix(y_true, y_pred)
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
        ax_cm.set_xlabel("Predicción")
        ax_cm.set_ylabel("Real")
        st.pyplot(fig_cm)

        st.subheader("Reporte de Clasificación")
        report = classification_report(y_true, y_pred, output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        st.dataframe(df_report.style.format("{:.2f}"))
    else:
        st.info("No se ha encontrado la columna 'score_model' y 'loan_status'. Por favor, entrená un modelo y añadí los scores.")

# Conclusiones
with tabs[3]:
    st.header("📌 Conclusiones del TFM")
    st.markdown("""
    - El modelo permite estimar el riesgo crediticio con buena precisión.
    - Variables como el ingreso, monto del préstamo y score de crédito muestran correlaciones clave con la probabilidad de impago.
    - Se sugiere aplicar el modelo en entornos reales con actualizaciones periódicas.
    - La visualización ayuda a detectar perfiles de riesgo y tomar decisiones informadas.
    """)

st.caption("TFM - Paula S. · 2025 · Universidad XYZ")