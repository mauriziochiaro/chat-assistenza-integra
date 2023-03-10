from langchain.prompts.prompt import PromptTemplate
from langchain.llms import OpenAI
# from langchain.chat_models import ChatOpenAI
from langchain.chains import ChatVectorDBChain
import streamlit as st

openai_api_key = st.secrets["OPENAI_API_KEY"]

_template = """Data la seguente conversazione e una domanda di follow-up, riformula la domanda di follow-up in una domanda autonoma.

Conversazione:
{chat_history}
Follow Up Input: {question}
Domanda autonoma:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

template = """Sei un assistente AI e sai rispondere a domande tecniche sull'uso del software Progetto INTEGRA.
Ti vengono fornite le seguenti parti estratte da un lungo documento e una domanda. Fornisci una risposta colloquiale.
Se non conosci la risposta, dì semplicemente "Hmm, non ne sono sicuro". Non cercare di inventare una risposta.
Se la domanda non riguarda Progetto INTEGRA, informali gentilmente che sei istruito solo per rispondere a domande su Progetto INTEGRA.
Domanda: {question}
=========
{context}
=========
Fornisci la risposta in formato Markdown:"""
QA_PROMPT = PromptTemplate(template=template, input_variables=["question", "context"])


def get_chain(vectorstore):
    llm = OpenAI(temperature=0, max_tokens=1000, openai_api_key=openai_api_key)
    # llm = OpenAI(model="gpt-3.5-turbo", temperature=0, max_tokens=1000)
    qa_chain = ChatVectorDBChain.from_llm(
        llm,
        vectorstore,
        qa_prompt=QA_PROMPT,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
    )
    return qa_chain
