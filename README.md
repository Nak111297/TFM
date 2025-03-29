# 💼 Evaluador de Riesgo Crediticio

Aplicación interactiva construida con Streamlit que permite evaluar el riesgo crediticio de un solicitante, mostrar una predicción de impago y explicar las razones detrás de dicha predicción utilizando interpretabilidad basada en SHAP.

---

## 🚀 ¿Qué hace esta app?

- Recibe datos financieros y personales de un solicitante.
- Estima la probabilidad de impago utilizando un modelo entrenado con XGBoost.
- Muestra una explicación SHAP en forma de gráfico waterfall.
- Genera una explicación en lenguaje natural usando IA (requiere token de HuggingFace).

---

## ⚙️ Instalación

### 1. Clonar el repositorio o copiar los archivos

Asegúrate de tener:
- `app_riesgo_crediticio.py`
- `xgb_model_credit_risk.pkl`
- `requirements.txt`

### 2. Crear un entorno virtual (opcional)

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate   # Windows
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## ▶️ Ejecutar la app

```bash
streamlit run app_riesgo_crediticio.py
```

---

## 🔐 Configuración del Token de Hugging Face (opcional)

Para habilitar la explicación generada por IA, necesitas una cuenta gratuita en [Hugging Face](https://huggingface.co) y un token.

```bash
export HF_TOKEN=tu_token_aqui  # Linux/Mac
set HF_TOKEN=tu_token_aqui     # Windows
```

---

## 📊 Modelo

- Algoritmo: `XGBoost`
- Métricas de evaluación: `Precision`, `Recall`, `F1-score`, `AUC-ROC`
- Variables seleccionadas tras ingeniería de características
- El modelo fue entrenado para predecir la probabilidad de impago (clase 1)

---

## 🧠 Interpretabilidad

- Se usa SHAP para entender la contribución de cada variable a la predicción individual.
- Se muestra un gráfico `waterfall` de SHAP.
- Se genera una explicación en lenguaje natural con Mistral-7B (opcional).

---