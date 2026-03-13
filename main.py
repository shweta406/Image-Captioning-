import streamlit as st
import numpy as np
import tempfile
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pickle


@st.cache_resource
def load_models():
    """Load models and tokenizer once and cache them for the session."""
    caption_model = load_model("model.keras")
    feature_extractor = load_model("feature_extractor.keras")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return caption_model, feature_extractor, tokenizer


def generate_caption(image_path, caption_model, feature_extractor, tokenizer, max_length=34, img_size=224):
    """Extract image features and generate a caption word by word."""
    img = load_img(image_path, target_size=(img_size, img_size))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    image_features = feature_extractor.predict(img_array, verbose=0)

    in_text = "startseq"
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = caption_model.predict([image_features, sequence], verbose=0)
        word = tokenizer.index_word.get(np.argmax(yhat), None)
        if word is None or word == "endseq":
            break
        in_text += " " + word

    caption = in_text.replace("startseq", "").strip()
    return caption


def main():
    st.set_page_config(page_title="Image Caption Generator", page_icon="🖼️", layout="centered")
    st.title("🖼️ Image Caption Generator")
    st.write("Upload an image and the model will generate a descriptive caption for it.")

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        with st.spinner("Loading models (first run may take a moment)..."):
            caption_model, feature_extractor, tokenizer = load_models()

        # Write upload to a temp file so Keras can read it from disk
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(uploaded_image.getbuffer())
            tmp_path = tmp_file.name

        try:
            with st.spinner("Generating caption..."):
                caption = generate_caption(tmp_path, caption_model, feature_extractor, tokenizer)

            st.image(uploaded_image, use_container_width=True)
            st.markdown("### Generated Caption")
            st.success(caption)
        finally:
            os.unlink(tmp_path)


if __name__ == "__main__":
    main()
