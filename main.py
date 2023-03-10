"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message
from langchain.embeddings import OpenAIEmbeddings
OpenAIEmbeddings.max_retries=6
# from ingest_data import embed_doc

from query_data import _template, CONDENSE_QUESTION_PROMPT, QA_PROMPT, get_chain

import pickle
import os

# From here down is all the StreamLit UI.
st.set_page_config(page_title="Assistenza INTEGRA beta", page_icon=":robot:")
st.header("Assistenza INTEGRA beta")

# Use the file uploader to get the path to the vectorstore file
uploaded_file = st.file_uploader("Upload your vectorstore file", type=["pkl"])

# Check if a file has been uploaded
if uploaded_file is not None:
    # Load the vectorstore from the uploaded file
    vectorstore = pickle.load(uploaded_file)
    chain = get_chain(vectorstore)
    st.success("Vectorstore loaded successfully!")   

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

placeholder = st.empty()
def get_text():
    input_text = placeholder.text_input("You: ", value="", key="input")
    return input_text


user_input = get_text()
print(st.session_state.input)
print(user_input)

if user_input:
    docs = vectorstore.similarity_search(user_input)
    print(len(docs))
    output = chain.run(input=user_input,vectorstore=vectorstore,context=docs[:2],chat_history = [],question=user_input,QA_PROMPT=QA_PROMPT,CONDENSE_QUESTION_PROMPT=CONDENSE_QUESTION_PROMPT,template=_template)

    st.session_state.past.append(user_input)
    print(st.session_state.past)
    st.session_state.generated.append(output)
    print(st.session_state.past)

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
