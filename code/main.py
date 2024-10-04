
import os
import pandas as pd
import chromadb
from getpass import getpass
from langchain_openai import OpenAI
from fastapi import FastAPI
from pydantic import BaseModel
from uuid import uuid4
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

data_folder_dir = '../../data'
HUGGINGFACEHUB_API_TOKEN = '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'

os.environ['HUGGINGFACEHUB_API_TOKEN'] = HUGGINGFACEHUB_API_TOKEN

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': False, 'clean_up_tokenization_spaces': True}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="/project/data/chroma_langchain_db",  # Where to save data locally, remove if not necessary
)

app = FastAPI()

class QueryRequest(BaseModel):
    text_body: str

@app.post("/query_vectordb")
def query_vectordb(request: QueryRequest):
    results = vector_store.similarity_search(
        request.text_body,
        k=2,
    )
    return results