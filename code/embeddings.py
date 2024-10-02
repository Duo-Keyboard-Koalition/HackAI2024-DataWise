import os
from getpass import getpass
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import CSVLoader

HUGGINGFACEHUB_API_TOKEN = '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'

os.environ['HUGGINGFACEHUB_API_TOKEN'] = HUGGINGFACEHUB_API_TOKEN

embeddings = HuggingFaceEmbeddings()

loader = CSVLoader('test.csv')

data = loader.load()

embedded_doc = embeddings.embed_documents(text.page_content for text in data)
print(embedded_doc)