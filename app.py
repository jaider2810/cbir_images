import os
import tempfile
import numpy as np
import streamlit as st
import cv2
import joblib
from descripteurs import concat_feat
from distances import Recherche_Image_Similaire

st.set_page_config(layout="wide")
st.title("CBIR Hybride avec Prédiction (KNN)")

distance = st.selectbox(
    "Distance",
    ["euclidienne", "manhattan", "chebychev", "canberra"]
)
k = st.slider("Top K", 1, 20, 5)

uploaded = st.file_uploader(
    "Image requête",
    type=["jpg", "png", "jpeg"]
)

if uploaded:
    file_bytes = np.asarray(
        bytearray(uploaded.read()), dtype=np.uint8
    )
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        width=300
    )

    fd, tmp = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    cv2.imwrite(tmp, img)

    query = concat_feat(tmp)
    query = np.array(query).reshape(1, -1)

    scaler = joblib.load("scaler.joblib")
    query_scaled = scaler.transform(query)

    model = joblib.load("best_model.joblib")
    predicted_class = model.predict(query_scaled)[0]

    st.subheader("Classe prédite")
    st.write(predicted_class)

    signatures = np.load(
        "Signatures_Concat.npy", allow_pickle=True
    )

    filtered = [
        s for s in signatures if s[-2] == predicted_class
    ]
    filtered = np.array(filtered, dtype=object)

    query_feat = query_scaled.flatten().tolist()

    results = Recherche_Image_Similaire(
        filtered, distance, query_feat, k
    )

    cols = st.columns(5)
    for i, (path, d, label) in enumerate(results):
        with cols[i % 5]:
            st.image(
                os.path.join("dataset", path),
                use_container_width=True
            )
            st.write(label)
            st.write(d)
