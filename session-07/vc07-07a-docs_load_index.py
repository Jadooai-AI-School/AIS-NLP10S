import os
import glob
import warnings
from typing import List
from langchain.schema import Document
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    TextLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader
)

warnings.simplefilter(action='ignore', category=FutureWarning)
from pathlib import Path
os.chdir(Path(__file__).resolve().parent)  # Setting the working directory to the script's location
data_path = Path("data")  # Defining the data path
# ============================================================================================
def load_documents_from_mixed_folder(directory: str) -> List[Document]:
    """Loads all supported documents from a folder, regardless of extension."""

    loader_classes = {
        ".pdf": PyMuPDFLoader,
        ".txt": TextLoader,
        ".csv": CSVLoader,
        ".docx": UnstructuredWordDocumentLoader,
        ".pptx": UnstructuredPowerPointLoader,
    }

    supported_extensions = set(loader_classes.keys())
    documents = []

    for filepath in glob.glob(os.path.join(directory, "**/*.*"), recursive=True):
        ext = os.path.splitext(filepath)[1].lower()
        if ext not in supported_extensions:
            print(f"Skipping unsupported file: {filepath}")
            continue

        loader = loader_classes[ext]
        try:
            docs = loader(filepath).load()
            for doc in docs:
                doc.metadata["source"] = os.path.basename(filepath)
                doc.metadata["type"] = ext[1:]  # remove the dot

                if ext in [".pdf", ".docx", ".pptx"]:
                    doc.metadata["page"] = list(range(1, len(doc.page_content.split("\f")) + 1))
                documents.append(doc)
            print(f"Loaded {filepath}")
        except Exception as e:
            print(f"Failed to load {filepath}: {e}")

    if not documents:
        print("No documents found.")
    else:
        print(f"Total documents loaded: {len(documents)}")
    return documents

# ============================================================================================
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

db_path = data_path/'index.db'

def create_index(documents: List[Document], split_size: int, overlap: int, index_name='my_index', persist: bool = True) -> FAISS:
    """Splits documents into chunks, creates a FAISS index, and stores it."""
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=split_size, chunk_overlap=overlap)
    split_docs = text_splitter.split_documents(documents)
    print("Total splits:", len(split_docs))

    embeddings_api = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device}
    )

    vector_index_store = FAISS.from_documents(split_docs, embeddings_api)

    if persist:
        vector_index_store.save_local(db_path, index_name)
        print(f"Index saved to: {db_path}")
    
    return vector_index_store

# ============================================================================================
def load_index_db(index_name: str = 'my_index') -> FAISS:
    """Loads an existing FAISS index from disk."""
    
    if os.path.exists(db_path):
        embeddings_api = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": device}
        )

        return FAISS.load_local(
            db_path,
            index_name=index_name,
            embeddings=embeddings_api,
            allow_dangerous_deserialization=True
        )
    else:
        print(f"Index not found: {db_path}")
        return None

# ============================================================================================
if __name__ == "__main__":
    os.system("clear" if os.name == "posix" else "cls")

    folder_path = data_path  # Adjust as needed

    documents = load_documents_from_mixed_folder(folder_path)

    if documents:
        create_index(documents, split_size=1000, overlap=100, index_name="my_index", persist=True)
