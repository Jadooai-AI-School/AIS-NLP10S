# Importing necessary modules and functions
from ais_rag.loader import load_documents  # Module for loading documents from the given path
from ais_rag.cleaner import clean_documents  # Module for cleaning and preprocessing the documents
from ais_rag.chunker import RecursiveTextSplitter  # Module for splitting documents into chunks
from ais_rag.vectordb import FAISSDB  # FAISS-based indexer for vectorizing and storing documents
import warnings
warnings.filterwarnings("ignore")
import os
from pathlib import Path
os.system('cls' if os.name == 'nt' else 'clear')

os.chdir(Path(__file__).parent)
print("Current working directory:", os.getcwd())
# Setting the path to the directory containing documents
# This path should contain the documents to be processed
path = 'data'

# Loading documents from the specified path
docs = load_documents(path)

# Cleaning the loaded documents
clean_documents = clean_documents(docs)

# Initializing the text splitter for chunking documents
splitter = RecursiveTextSplitter()

# Splitting cleaned documents into chunks using the specified strategy
chunked_documents = splitter.split_documents(documents=clean_documents, strategy='sentence')

# Initializing FAISS vector store with a specific pre-trained model
faiss_db = FAISSDB(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Building the FAISS index using the chunked documents
faiss_db.build_index(chunked_documents)