import os
import tempfile
import numpy as np
import streamlit as st
from PIL import Image
from descripteurs import concat_feat
from distances import Recherche_Image_Similaire

st.set_page_config(layout="wide")
st.title("CBIR avec Streamlit")

distance = st.selectbox("Distance", ["euclidienne", "manhattan", "chebychev", "canberra"])
k = st.slider("Top K", 1, 20, 5)

uploaded = st.file_uploader("Image requête", type=["jpg", "png", "jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, width=300)

    fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    img.save(tmp_path)

    query = concat_feat(tmp_path)

    signatures = np.load("Signatures_Concat.npy", allow_pickle=True)

    results = Recherche_Image_Similaire(signatures, distance, query, k)

    cols = st.columns(5)
    for i, (path, dist, label) in enumerate(results):
        full_path = os.path.join("dataset", path)
        with cols[i % 5]:
            st.image(full_path, use_container_width=True)
            st.write(label)
            st.write(dist)
