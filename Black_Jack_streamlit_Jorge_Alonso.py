import streamlit as st
import pandas as pd
import joblib
import numpy as np


# Cargar datasets
ds_blackjack = pd.read_csv("blkjckhands.csv", nrows=200004)

#1 - LIMPIEZA Y TRASFORMACIÓN DE DATOS

#Eliminamos columnas referente a apuestas
ds_blackjack = ds_blackjack.drop(["plwinamt", "dlwinamt"], axis=1)

# Suma dos primeras cartas del jugador
ds_blackjack = ds_blackjack.rename(columns= {"player_2cards_sum": "ply2cardsum"})

# número de cartas totales pedidas por el jugador (cuenta en una única fila, el numero de cartas que ha pedido el jugador)
ds_blackjack["ply_No_cards"] = ds_blackjack[["card1", "card2", "card3", "card4", "card5"]].ne(0).sum(axis=1) #.e(0) quiere decir not equal 0

# número de cartas totales pedidas por el dealer (cuenta en una única fila, el numero de cartas que ha pedido el dealer)
ds_blackjack["deal_No_cards"] = ds_blackjack[["dealcard1", "dealcard2", "dealcard3", "dealcard4", "dealcard5"]].ne(0).sum(axis=1)

# Suma total de cartas del dealer en la mesa al empezar la partida (player vs crepier)
ds_blackjack["deal_2cards_sum"] = ds_blackjack[["dealcard1", "dealcard2"]].sum(axis=1)

# Suma total de cartas visibles en la mesa al empezar la partida (player vs crepier)
ds_blackjack["sum_3first_cards"] = ds_blackjack[["card1", "card2", "dealcard1"]].sum(axis=1)

#trasformar las variables categoricas en numericas, la mas importante win, los, push
def  numeric_def (x):
    if x == "Loss":
        return 0
    if x == "Push":
        return 1
    if x == "Win":
        return 2
    
ds_blackjack["winloss_numeric"] = ds_blackjack["winloss"].apply(numeric_def)

# Creamos la función para pasar la fencuin black jack a numerica y la aplicamos en un nueva columna
def blkjck_def (x):
    if x == "nowin":
        return 0
    if x == "Win":
        return 1    
ds_blackjack["blkjck_numeric"] = ds_blackjack["blkjck"].apply(blkjck_def)

# Creamos la columna plybustbeat en formato numerica
def plybustbeat_def (x):
    if x == "Push":
        return 0
    if x == "Plwin":
        return 1
    if x == "DlBust":
        return 2
    if x == "Beat":
        return 3
    if x == "Bust":
        return 4    
ds_blackjack["plybustbeat_numeric"] = ds_blackjack["plybustbeat"].apply(plybustbeat_def)


## FIN DE LA LIMPIEZA Y TRASFORMACIÓN DE LOS DATOS

# 2 - CREACIÓN Y REPRESENTACIÓN DE LOS GRÁFICOS

#-----------------------------

st.title("Black Jack - Asistente virtual")

st.image("bj.jpg", caption="ML aplicado a Blackjack -  Jorge Alonso", use_container_width=True)


#---------------------------------------------------------------------MODELO DE PREDICCIÓN-------------------------------------------------------------

#importamos los archivos pkl de grid_cv y model

grid_cv = joblib.load("grid_cv.pkl")
model = joblib.load("model.pkl")

#Importamos el Witget de predicción del modelo final y lo adaptamos a streamlit


# Diccionario de resultados posibles en la partida
label_map = {0: "Loss", 1: "Push", 2: "Win"}

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

# Encabezado para streamlit
st.header("Predicción Blackjack")

# Inputs (equivalentes a tus IntText)
col1, col2 = st.columns(2)
with col1:
    w_sumofcards   = st.number_input("Suma cartas del jugador:",  min_value=0, step=1, value=10)
    w_ply2cardsum  = st.number_input("Suma 2 primeras cartas del jugador:", min_value=0, step=1, value=10)
with col2:
    w_dealcard1    = st.number_input("Carta visibe del dealer:",   min_value=0, step=1, value=5)
    w_ply_No_cards = st.number_input("Número de cartas del jugador:",min_value=0, step=1, value=2)

if st.button("Predecir", type="primary"):
    # Construir DataFrame igual que en tu notebook
    X_manual = pd.DataFrame([{
        "sumofcards": w_sumofcards,
        "dealcard1": w_dealcard1,
        "ply2cardsum": w_ply2cardsum,
        "ply_No_cards": w_ply_No_cards
    }])

    # Predicción
    pred = model.predict(X_manual)[0]
    st.write(f"**Predicción:** {label_map[int(pred)]}  *(0=Loss, 1=Push, 2=Win)*")

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
