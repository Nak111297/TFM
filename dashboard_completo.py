import streamlit as st
import pandas as pd
import plotly.express as px

# Configuración de página
st.set_page_config(page_title="Dashboard con filtros", layout="wide")
st.title("📊 Dashboard del Dataset Original con Filtros")

# Cargar dataset original
@st.cache_data
def cargar_datos():
    return pd.read_csv("loan_data.csv")

df = cargar_datos()

# Sidebar - Filtros interactivos
st.sidebar.header("🎛️ Filtros")
if 'person_age' in df.columns:
    edad = st.sidebar.slider("Edad", int(df['person_age'].min()), int(df['person_age'].max()), (int(df['person_age'].min()), int(df['person_age'].max())))
if 'person_income' in df.columns:
    ingreso = st.sidebar.slider("Ingreso", int(df['person_income'].min()), int(df['person_income'].max()), (int(df['person_income'].min()), int(df['person_income'].max())))
if 'loan_amnt' in df.columns:
    monto = st.sidebar.slider("Monto del Préstamo", int(df['loan_amnt'].min()), int(df['loan_amnt'].max()), (int(df['loan_amnt'].min()), int(df['loan_amnt'].max())))
if 'credit_score' in df.columns:
    score = st.sidebar.slider("Credit Score", int(df['credit_score'].min()), int(df['credit_score'].max()), (int(df['credit_score'].min()), int(df['credit_score'].max())))
if 'loan_int_rate' in df.columns:
    int_rate = st.sidebar.slider("Tasa de Interés", float(df['loan_int_rate'].min()), float(df['loan_int_rate'].max()), (float(df['loan_int_rate'].min()), float(df['loan_int_rate'].max())))

# Aplicar filtros
columnas_requeridas = ['person_age', 'person_income', 'loan_amnt', 'credit_score', 'loan_int_rate']
if all(col in df.columns for col in columnas_requeridas):
    df = df[
        df['person_age'].between(*edad) &
        df['person_income'].between(*ingreso) &
        df['loan_amnt'].between(*monto) &
        df['credit_score'].between(*score) &
        df['loan_int_rate'].between(*int_rate)
    ]

# KPIs generales
st.subheader("📌 Estadísticas generales del dataset filtrado")
col1, col2, col3 = st.columns(3)
col1.metric("Registros totales", len(df))
col2.metric("Edad promedio", round(df['person_age'].mean(), 1) if 'person_age' in df.columns else "-")
col3.metric("Ingreso promedio", f"${round(df['person_income'].mean(), 2):,.0f}" if 'person_income' in df.columns else "-")

# Visualizaciones principales
st.subheader("📊 Distribución de variables principales")
col4, col5 = st.columns(2)
if 'loan_amnt' in df.columns:
    with col4:
        fig1 = px.histogram(df, x='loan_amnt', nbins=30, title="Distribución del monto del préstamo")
        st.plotly_chart(fig1, use_container_width=True)
if 'loan_int_rate' in df.columns:
    with col5:
        fig2 = px.histogram(df, x='loan_int_rate', nbins=30, title="Distribución de tasa de interés")
        st.plotly_chart(fig2, use_container_width=True)

# Relación ingreso-score si están disponibles
if 'person_income' in df.columns and 'credit_score' in df.columns:
    st.subheader("🎯 Relación entre ingreso y score de crédito")
    fig3 = px.scatter(df, x='person_income', y='credit_score',
                      title="Ingreso vs Credit Score")
    st.plotly_chart(fig3, use_container_width=True)

st.caption("Este dashboard muestra el dataset original con capacidad de filtrado interactivo.")