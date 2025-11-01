import sys
import os
import streamlit as st
import ollama

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_dir)

from src.main import multimodal_pipeline
from src.input_processing import speech_to_text  # whisper transcription

# streamlit app configuration
st.set_page_config(page_title="multimodal rag assistant", page_icon=None)
st.title("multimodal rag assistant")

# input selection
input_type = st.selectbox("select input type", ["text", "audio", "image"])
user_input = None

if input_type == "text":
    user_input = st.text_area("enter your query:")
elif input_type == "audio":
    user_input = st.file_uploader("upload audio file", type=["wav", "mp3", "m4a"])
elif input_type == "image":
    user_input = st.file_uploader("upload image", type=["jpg", "jpeg", "png"])

# generate response
if st.button("generate"):
    if user_input:
# audio input
        if input_type == "audio":
            with st.spinner("transcribing audio..."):
                transcription, success = speech_to_text(user_input)

            if success:
                st.write("transcription:", transcription)
            else:
                st.write("audio processing failed.")
            st.stop()


        else:
            with st.spinner("processing"):
                query, answer, context = multimodal_pipeline(
                    input_type, user_input, return_context=True
                )

            st.write("response:")
            st.write(answer)

            st.write("context:")
            st.write(context)
    else:
        st.write("No INPUT")
