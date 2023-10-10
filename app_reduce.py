from fastapi import FastAPI
from langchain.chains import RetrievalQA
# from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import ElasticVectorSearch
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import AzureOpenAI
from langchain.prompts import PromptTemplate
from langchain.vectorstores.elasticsearch import ElasticsearchStore
from langchain.chains.query_constructor.base import AttributeInfo


import os
import streamlit as st
import openai

from config import openai_api_key


embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')


HUGGING_FACE_API_KEY="xxxx"

elastic_host = "xxxx.es.europe-west1.gcp.cloud.es.io"
elasticsearch_url = f"https://elastic:xxxx@{elastic_host}:9243"

db = ElasticsearchStore(
    es_url=elasticsearch_url,
    index_name="search-llama-750g",
    embedding=embedding,
    es_user="elastic",
    es_password="xxxx",
    query_field="text",
    vector_query_field="vector"
)

openai.api_key = 'YOU_KEY'
openai.api_base = 'YOUR_API_BASE' # your endpoint should look like the following https://YOUR_RESOURCE_NAME.openai.azure.com/
openai.api_type = 'azure'
openai.api_version = '2023-05-15' # this may change in the future
deployment_name='YOUR_DEPLOYMENT_NAME'


prompt_template_loose = """Utilisez les éléments de contexte suivants pour répondre à la question à la fin. Si vous ne connaissez pas la réponse,  inventez une réponse.
{context}
Question: {question}
Réponse en français:"""

prompt_template_strict = """Utilisez les éléments de contexte suivants pour répondre à la question à la fin. Si vous ne connaissez pas la réponse,  dites simplement que vous ne savez pas, n'essayez pas d'inventer une réponse.
{context}
Question: {question}
Réponse en français:"""

PROMPT = PromptTemplate(
    template=prompt_template_strict, input_variables=["context", "question"]
)
chain_type_kwargs = {"prompt": PROMPT}


qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(
        engine="huage-gpt-4",
        model_name="gpt-4-32k",
        temperature=0,
    ),
    chain_type="map_reduce",
    retriever=db.as_retriever(),
    return_source_documents=True,
    #chain_type_kwargs=chain_type_kwargs
)

app = FastAPI()

@app.post("/ask_recette")
def ask(query: str):
    # response = qa.run(query)
    response = qa({"query": query})
    #response = db.similarity_search(query, k=2)
    return {
        "response": response,
    }


st.title("Recette GPT Chunking Reduce")

# Main chat form
with st.form("chat_form"):
    querytest = st.text_input("Cherchez recette: ")
    submit_button = st.form_submit_button("Send")

# Generate and display response on form submission
if submit_button:
    resp = ask(querytest)
    st.write(f"ChatGPT: {resp['response']['result']}")
    st.write(f"Source: {resp['response']['source_documents'][0]}")


