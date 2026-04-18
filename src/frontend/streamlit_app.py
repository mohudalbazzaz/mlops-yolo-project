import streamlit as st
import requests


def run_ui() -> None:
    """
    Renders a simple Streamlit UI that allows users to upload an
    image of a banana, preview it, and request a ripeness prediction from
    the backend FastAPI service. The uploaded image is sent as multipart
    form data to the `/banana_ripeness_classifier` endpoint, and the
    resulting classification is displayed to the user.
    """
    st.title("Banana Ripeness Classifier")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            st.image(uploaded_file, caption="Uploaded Image", width=400)

        if st.button("Predict"):

            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type,
                )
            }

            try:
                response = requests.post(
                    "http://backend:8000/banana_ripeness_classifier", files=files
                )

                response.raise_for_status()

                result = response.json()
                prediction = result["result"]

                st.success(f"{prediction}")

            except Exception as e:
                st.error(e)


if __name__ == "__main__":
    run_ui()
