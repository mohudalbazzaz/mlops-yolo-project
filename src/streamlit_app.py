import streamlit as st
import requests

st.title("Banana Ripeness Classifier 🍌")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.image(uploaded_file, caption="Uploaded Image", width=400)

    if st.button("Predict"):

        files = {
            "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
        }

        response = requests.post(
            "http://127.0.0.1:8000/banana_ripeness_classifier",
            files=files
        )

        if response.status_code == 200:
            result = response.json()
            st.success(f"{result}")
        else:
            st.error("Error calling API")