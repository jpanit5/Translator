import streamlit as st
import cv2
import pytesseract
from PIL import Image
import numpy as np
from deep_translator import GoogleTranslator

st.set_page_config(page_title="Image Translator", layout="centered")

st.title("🌍 Image Translator (FREE)")
st.caption("Upload image → Extract → Translate (ENG → Chinese)")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if not uploaded_file:
    st.info("👆 Upload ka muna ng image")

if uploaded_file:
    image = Image.open(uploaded_file)
    img = np.array(image)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Processing..."):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # OCR
        text = pytesseract.image_to_string(
            gray,
            lang="eng+chi_sim",
            config="--oem 3 --psm 6"
        )

        cleaned_text = text.strip()

        col1, col2 = st.columns(2)

        # LEFT SIDE
        with col1:
            st.subheader("📝 Extracted Text")
            st.text(cleaned_text if cleaned_text else "No text detected")

        # RIGHT SIDE
        if cleaned_text:
            try:
                translated = GoogleTranslator(
                    source='auto',
                    target='zh-CN'  # ✅ FIXED HERE
                ).translate(cleaned_text)

            except Exception as e:
                translated = f"❌ Translation error: {e}"

            with col2:
                st.subheader("🌍 Translated Text")
                st.write(translated)

            st.download_button(
                "Download Translation",
                translated,
                file_name="translation.txt"
            )
