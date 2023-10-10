from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.document_loaders import UnstructuredHTMLLoader, BSHTMLLoader
from langchain.vectorstores.elasticsearch import ElasticsearchStore
from langchain.embeddings import HuggingFaceEmbeddings
import os

DATA_PATH="data/"


def create_vector_db():

    documents=[]
    processed_htmls=0
    processed_pdfs=0
    for f in os.listdir("web_data"):
        try:
            if f.endswith(".pdf"):
                pdf_path = 'YOU_PATH' + f
                loader = PyPDFLoader(pdf_path)
                documents.extend(loader.load())
                processed_pdfs+=1
            elif f.endswith(".html"):
                html_path = 'YOUR_PATH' + f
                print(html_path)
                loader = BSHTMLLoader(html_path)
                documents.extend(loader.load())
                processed_htmls+=1
        except:
            print("issue with ",f)
            pass
    print("Processed",processed_htmls,"html files and ",processed_pdfs,"pdf files")


    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts=text_splitter.split_documents(documents)

    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    elastic_host = "xxxx.es.europe-west1.gcp.cloud.es.io"
    elasticsearch_url = f"https://elastic:xxx@{elastic_host}:9243"

    db = ElasticsearchStore(es_url=elasticsearch_url,
        index_name="search-llama-750g",
        embedding=embeddings,
        es_user="elastic",
        es_password="xxx",
    )

    docs = db.from_documents(
        texts, embeddings, es_url=elasticsearch_url, es_user="elastic", ees_password="xxx",
        index_name="search-llama-750g"
    )


if __name__=="__main__":
    create_vector_db()