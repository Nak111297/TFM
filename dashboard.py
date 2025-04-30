import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import numpy as np

st.set_page_config(page_title="Riesgo Crediticio", layout="wide")
st.title("📊 Dashboard de Evaluación del Riesgo Crediticio")

# Cargar modelo y datos
modelo = pickle.load(open("xgb_model_credit_risk.pkl", "rb"))
df = pd.read_csv("loan_data.csv")

# Tabs principales
tab1, tab2 = st.tabs(["📈 Análisis Previo al Modelo", "🤖 Evaluación del Modelo"])

# ---------------- TAB 1: Análisis Previo ---------------- #
with tab1:
    st.header("Análisis de Impago por Variables Clave")

    # Variables importantes (puedes ajustar esta lista si tu modelo lo indica)
    variables_clave = [
        'person_income', 'person_age', 'loan_amnt', 'loan_int_rate',
        'credit_score', 'person_emp_exp', 'cb_person_cred_hist_length'
    ]

    var = st.selectbox("Selecciona una variable para analizar la tasa de default:", variables_clave)

    # Agrupar y calcular tasa de impago por grupos
    try:
        df_grouped = df[[var, 'loan_status']].copy()
        df_grouped[var] = pd.qcut(df_grouped[var], q=10, duplicates='drop')  # deciles
        df_grouped[var] = df_grouped[var].astype(str)  # convertir a texto para evitar error de serialización
        tasas = df_grouped.groupby(var)['loan_status'].mean().reset_index()
        tasas.columns = [var, 'Tasa de Default']

        fig = px.bar(tasas, x=var, y='Tasa de Default', title=f"Tasa de impago por {var}")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"No se pudo graficar: {e}")

# ---------------- TAB 2: Evaluación del Modelo ---------------- #
with tab2:
    st.header("Evaluación del Modelo de Machine Learning")

    if "score_model" in df.columns:
        y_true = df["loan_status"]
        y_score = df["score_model"]
        y_pred = (y_score >= 0.5).astype(int)

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        st.subheader("Curva ROC")
        fig_roc, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Curva ROC")
        ax.legend(loc='lower right')
        st.pyplot(fig_roc)

        # Distribución del score
        st.subheader("Distribución del Score de Riesgo")
        fig_hist = px.histogram(df, x="score_model", nbins=30, color=y_true.astype(str),
                                title="Score del modelo por clase real")
        st.plotly_chart(fig_hist, use_container_width=True)

        # Matriz de confusión
        st.subheader("Matriz de Confusión")
        cm = confusion_matrix(y_true, y_pred)
        fig_cm, ax_cm = plt.subplots()
        ax_cm.imshow(cm, cmap='Blues')
        ax_cm.set_title("Matriz de Confusión")
        ax_cm.set_xticks([0, 1])
        ax_cm.set_yticks([0, 1])
        ax_cm.set_xticklabels(['No Default', 'Default'])
        ax_cm.set_yticklabels(['No Default', 'Default'])
        ax_cm.set_xlabel('Predicción')
        ax_cm.set_ylabel('Real')

        for i in range(2):
            for j in range(2):
                ax_cm.text(j, i, cm[i, j], ha='center', va='center', color='black')

        st.pyplot(fig_cm)

        # Métricas
        st.subheader("Reporte de Clasificación")
        report = classification_report(y_true, y_pred, output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        st.dataframe(df_report.style.format("{:.2f}"))

    else:
        st.error("El dataset no contiene la columna 'score_model'. Añádela desde el modelo con `.predict_proba()` y guarda el archivo de nuevo.")

