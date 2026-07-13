"""Streamlit app for the spam/ham classifier.

Usage:
    streamlit run app/app.py
"""

import sys
from pathlib import Path

import streamlit as st

# Make src/ importable when running from the project root
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from predict import predict_email  # noqa: E402


def main():
    st.set_page_config(page_title="Spam/Ham Classifier", page_icon="📧")
    st.title("📧 Spam/Ham Classifier")
    st.write("Paste a message below to check whether it's spam or ham.")

    message = st.text_area("Message text", height=150)

    if st.button("Classify"):
        if not message.strip():
            st.warning("Please enter some text to classify.")
        else:
            with st.spinner("Classifying..."):
                result = predict_email(message)

            if result == "Spam":
                st.error(f"🚨 This message is **{result}**.")
            else:
                st.success(f"✅ This message is **{result}**.")


if __name__ == "__main__":
    main()
