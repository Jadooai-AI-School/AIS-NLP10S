import os
import glob
from langchain_community.document_loaders import (
    PyPDFLoader, PyMuPDFLoader,
    TextLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader
)
from langchain.schema import Document
from typing import Dict, List

def load_documents(directories: Dict[str, str]) -> List[Document]:
    """Loads documents from specified directories, extracting essential metadata.

    Args:
        directories: A dictionary mapping file types to their directory paths.

    Returns:
        A list of Document objects with file name, type, and page number metadata.
    """

    documents = []
    loader_classes = {
        "pdf": PyMuPDFLoader,
        "txt": TextLoader,
        "csv": CSVLoader,
        "docx": UnstructuredWordDocumentLoader,
        "pptx": UnstructuredPowerPointLoader
    }

    for file_type, directory in directories.items():
        loader = loader_classes.get(file_type.lower())
        if not loader:
            print(f"Unsupported file type: {file_type}")
            continue
        
        glob_pattern = os.path.join(directory, f"**/*.{file_type}")
        for filepath in glob.glob(glob_pattern, recursive=True):
            # print(filepath)
            docs = loader(filepath).load()
            for doc in docs:  # various document parts
                doc.metadata["source"] = os.path.basename(filepath)
                doc.metadata["type"] = file_type.lower()
                
                if file_type.lower() in ["pdf", "docx", "pptx"]:
                    # Calculate page numbers
                    doc.metadata["page"] = list(range(1, len(doc.page_content.split("\f")) + 1))
                documents.append(doc)
            print(f"Loaded {filepath}, type: {file_type}")

    return documents

####################################################################################################################################
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
import os
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device:', device)
db_path = './data/index.db'

def create_index(documents: List[Document], split_size: int, overlap: int, index_name = 'my_index', persist:bool =True) -> FAISS:
    """Splits documents into chunks, creates a FAISS index, and stores it.

    Args:
        documents: A list of Document objects.
        split_size: The desired size of each chunk (in characters).
        overlap: The number of overlapping characters between consecutive chunks.
        index_dbname: The path to store the FAISS index database.

    Returns:
        The path where the index database is stored.
    """
    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=split_size, chunk_overlap=overlap)

    # Split documents into chunks
    split_docs = text_splitter.split_documents(documents)
    print('Splits:', len(split_docs))

    # Creating embeddings
    embeddings_api = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device}  # Use GPU if available
        )
    
    # Creating a FAISS vector store from the preprocessed texts
    vector_index_store = FAISS.from_documents(split_docs, embeddings_api)

    # Save vectorstore
    if persist:
        vector_index_store.save_local(db_path, index_name)
        print(f"Index saved to: {db_path}")
    
    return vector_index_store


def load_index_db(index_name: str='my_index_default') -> FAISS:
    #Loads an existing FAISS index from disk.
    if os.path.exists(db_path):
        # Creating embeddings
        embeddings_api = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": device}  # Use GPU if available
            )
    
        return FAISS.load_local(db_path,
                                index_name=index_name,
                                embeddings=embeddings_api,
                                allow_dangerous_deserialization=True  # to suppress warning
                                )
    else:
        print(f"Index not found: {db_path}")
        return None
    
#######################################################################################################################################
if __name__ == "__main__":
    import os
    os.system("clear")
        
    doc_dict = {
        "pdf": "data/",
        "txt": "data/",
        "csv": "data/",
        "docx": "data/",
        "pptx": "data/"
        }
    
    documents = load_documents(doc_dict)

    create_index(documents, 1000, 100, "my_index", persist=True)

