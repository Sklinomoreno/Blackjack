import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Título de la app
st.title("Black Jack - Asistente virtual")
st.image("bj.jpg", caption="ML aplicado a Blackjack")

# Cargar los archivos del modelo
grid_cv = joblib.load("grid_cv.pkl")
model = joblib.load("model.pkl")

# Diccionario de resultados posibles en la partida
label_map = {0: "Loss", 1: "Push", 2: "Win"}

# Función para obtener probabilidades
def get_probabilities_safe(model, X):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        classes = getattr(model, "classes_", None)
        return proba, classes, False
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
        scores = scores - scores.max(axis=1, keepdims=True)
        exp_s = np.exp(scores)
        proba = exp_s / exp_s.sum(axis=1, keepdims=True)
        classes = getattr(model, "classes_", np.arange(proba.shape[1]))
        return proba, classes, True
    classes = getattr(model, "classes_", np.array([0, 1, 2]))
    n_classes = len(classes)
    proba = np.full((len(X), n_classes), 1.0 / n_classes)
    return proba, classes, True

# Encabezado para la predicción
st.header("Predicción Blackjack")

# Inputs para la predicción
col1, col2 = st.columns(2)
with col1:
    w_sumofcards = st.number_input("Suma cartas del jugador:", min_value=0, step=1, value=10)
    w_ply2cardsum = st.number_input("Suma 2 primeras cartas del jugador:", min_value=0, step=1, value=10)
with col2:
    w_dealcard1 = st.number_input("Carta visible del dealer:", min_value=0, step=1, value=5)
    w_ply_No_cards = st.number_input("Número de cartas del jugador:", min_value=0, step=1, value=2)

# Botón para predecir
if st.button("Predecir", type="primary"):
    # Construir DataFrame para la predicción
    X_manual = pd.DataFrame([{
        "sumofcards": w_sumofcards,
        "dealcard1": w_dealcard1,
        "ply2cardsum": w_ply2cardsum,
        "ply_No_cards": w_ply_No_cards
    }])
    
    # Predicción
    pred = model.predict(X_manual)[0]
    st.write(f"**Predicción:** {label_map[int(pred)]} *(0=Loss, 1=Push, 2=Win)*")
    
    # Probabilidades
    proba, classes, aproximadas = get_probabilities_safe(model, X_manual)
    classes = np.asarray(classes).astype(int)
    labels = [label_map.get(c, str(c)) for c in classes]
    perc = (proba[0] * 100).round(2)
    perc_dict = dict(zip(labels, perc))
    ordered_cols = ["Loss", "Push", "Win"]
    df_proba = pd.DataFrame([{col: perc_dict.get(col, 0.0) for col in ordered_cols}])
    titulo = "Probabilidades por clase (%)"
    if aproximadas:
        titulo += " (aprox.)"
    st.subheader(titulo)
    st.dataframe(df_proba.style.format("{:.2f}%"), use_container_width=True)
    
    # Recomendación según % Loss
    loss_pct = perc_dict.get("Loss", 0.0)
    if loss_pct > 50:
        st.success("♣️ **Recomendación:** Pedir carta")
    else:
        st.info("♣️ **Recomendación:** Plantarse")
