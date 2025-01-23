import streamlit as st
from utils import apply_styles
from DataScientist import *

st.title("Your Data Scientist")

if st.button("💬 New Chat"):
  st.session_state.messages = []
  st.rerun()

apply_styles()

if "messages" not in st.session_state:
  st.session_state.messages = []

for message in st.session_state.messages:
  with st.chat_message(message["role"]):
    st.markdown(message["content"])

if prompt := st.chat_input("How are you doing Babe?"):
  st.session_state.messages.append({"role": "Arun", "content": prompt})
  with st.chat_message("user"):
    st.markdown(prompt)

  with st.chat_message("assistant"):
    chunks = DSAnalyst.run(prompt, stream=True)
    response = st.write_stream(as_stream(chunks))
  st.session_state.messages.append({"role": "Data Scientist", "content": response})