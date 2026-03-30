import streamlit as st
import cv2
import pytesseract
from PIL import Image
import numpy as np
from deep_translator import GoogleTranslator
from streamlit_paste_button import paste_image_button

st.set_page_config(page_title="Image Translator", layout="centered")

st.title("Image Translator")
st.caption("Paste Image, Upload, or Paste Text → Translate (English to Chinese)")

# ==============================
# CACHE FUNCTION (MUST BE TOP LEVEL)
# ==============================
@st.cache_data(show_spinner=False)
def run_ocr(gray_img):
    return pytesseract.image_to_string(
        gray_img,
        lang="eng+chi_sim",
        config="--oem 3 --psm 6"
    )

# ==============================
# SESSION STATE
# ==============================
if "image" not in st.session_state:
    st.session_state.image = None

# ==============================
# TEXT INPUT
# ==============================
st.subheader("Text Input (Optional)")
text_input = st.text_area("Paste text here")

# ==============================
# IMAGE INPUT
# ==============================
paste_result = paste_image_button("Paste Image")
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

# Handle new inputs
if paste_result.image_data is not None:
    st.session_state.image = paste_result.image_data
elif uploaded_file is not None:
    st.session_state.image = Image.open(uploaded_file)

image = st.session_state.image

# Clear button
if image is not None:
    if st.button("Clear Image"):
        st.session_state.image = None
        st.rerun()

# ==============================
# MAIN LOGIC (FIXED STRUCTURE)
# ==============================
if text_input.strip():
    # TEXT PRIORITY
    st.subheader("Input Text")
    st.write(text_input)

    with st.spinner("Translating..."):
        try:
            translated = GoogleTranslator(
                source="auto",
                target="zh-CN"
            ).translate(text_input)
        except Exception as e:
            translated = f"Translation error: {e}"

    st.subheader("Translated Text")
    st.write(translated)

elif image is not None:
    # IMAGE PROCESSING
    img = np.array(image)

    # Resize for performance
    max_width = 800
    scale = max_width / img.shape[1]
    new_height = int(img.shape[0] * scale)
    img = cv2.resize(img, (max_width, new_height))

    st.image(image, caption="Input Image", width="stretch")

    with st.spinner("Processing image..."):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        text = run_ocr(gray)
        cleaned_text = text.strip()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Extracted Text")
            st.text(cleaned_text if cleaned_text else "No text detected")

        if cleaned_text:
            try:
                translated = GoogleTranslator(
                    source="auto",
                    target="zh-CN"
                ).translate(cleaned_text)
            except Exception as e:
                translated = f"Translation error: {e}"

            with col2:
                st.subheader("Translated Text")
                st.write(translated)

            st.download_button(
                "Download Translation",
                translated,
                file_name="translation.txt"
            )

else:
    # NO INPUT
    st.info("Provide input by pasting text, pasting an image, or uploading a file.")
