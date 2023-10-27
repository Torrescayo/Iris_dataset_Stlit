import streamlit as st
import joblib
import numpy as np

# Cargar modelos y pipeline
pipeline = joblib.load('/home/torrescayo/ML/pipeline_svm_lr.sav')
svm_model = joblib.load('/home/torrescayo/ML/svm_model.sav')
lr_model = joblib.load('/home/torrescayo/ML/lr_model.sav')
dt_model = joblib.load('/home/torrescayo/ML/dt_model.sav')

def get_prediction(model, features):
    if model in ['SVM', 'Logistic Regression']:
        features = pipeline.transform(features)
    prediction = model.predict(features)
    return prediction

st.title("Predicción de tipos de flores Iris")

# Recoger características del usuario
sepal_length = st.number_input('Longitud del sépalo (cm)', min_value=0.0, max_value=10.0)
sepal_width = st.number_input('Ancho del sépalo (cm)', min_value=0.0, max_value=10.0)
petal_length = st.number_input('Longitud del pétalo (cm)', min_value=0.0, max_value=10.0)
petal_width = st.number_input('Ancho del pétalo (cm)', min_value=0.0, max_value=10.0)

# Seleccionar modelo
model_choice = st.selectbox("Selecciona el modelo", ["SVM", "Logistic Regression", "Decision Tree"])

# Mapeo de etiquetas numéricas a nombres
class_map = {
    0: "Iris Setosa",
    1: "Iris Versicolor",
    2: "Iris Virginica"
}

# Botón para realizar la predicción
if st.button("Predecir"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    if model_choice == "SVM":
        prediction = get_prediction(svm_model, features)
    elif model_choice == "Logistic Regression":
        prediction = get_prediction(lr_model, features)
    else:
        prediction = get_prediction(dt_model, features)

    st.write(f"La flor es de tipo: {class_map[prediction[0]]}")

