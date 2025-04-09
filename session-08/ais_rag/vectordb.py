import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import os
import logging
import json
from tqdm import tqdm

# Configuring logging
logging.basicConfig(level=logging.INFO)

# Defining the device based on availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Document:
    def __init__(self, page_content: str, metadata: dict):
        # Initializing Document with page content and metadata
        self.page_content = page_content
        self.metadata = metadata
        logging.info(f"Using device: {device}")

class FAISSDB:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        # Initializing FAISSDB with a sentence transformer model
        self.model = SentenceTransformer(model_name).to(device)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.metadata_file = 'metadata.json'  # Default file name for metadata

    def build_index(self, docs: List[Document], index_file_name='faiss.index', strategy='overwrite', index_type='ivf'):
        """Building or updating a FAISS index from a list of Document objects and managing metadata.

        Args:
            docs: List of Document objects to be indexed.
            index_file_name: The name of the index file.
            strategy: The indexing strategy ('overwrite' or 'append').
            index_type: The type of FAISS index to use ('flat', 'ivf').
        """
        self.metadata_file = f"{index_file_name}_metadata.json"

        ids = []
        vectors = []
        idx_doc = {}  # Dictionary to store metadata with index

        # Initializing progress bar
        logging.info("Creating vector database..")
        text_contents = [doc.page_content for doc in docs]
        embeddings = self.model.encode(text_contents, convert_to_tensor=True, device=device, show_progress_bar=True, batch_size=32)
        vectors = embeddings.cpu().numpy().astype(np.float32)
        ids = list(range(len(docs)))
        for i, doc in enumerate(docs):
            idx_doc[i] = {
                "idx": i,
                "content": doc.page_content,
                "metadata": doc.metadata
            }
        # Converting lists to numpy arrays
        ids = np.array(ids)
        vectors = np.vstack(vectors)  # Combining all vectors into a single 2D array

        # Creating or updating index
        logging.info("Creating Faiss index.")
        if index_type == 'ivf':
            nlist = 5  # Number of clusters
            quantizer = faiss.IndexFlatL2(self.dimension)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            index.train(vectors)
            index.nprobe = 2  # Number of clusters to search
        else:
            index = faiss.IndexFlatL2(self.dimension)  # Using L2 distance as the metric

        if strategy == 'overwrite':
            # Removing existing index file if present
            if os.path.exists(index_file_name):
                os.remove(index_file_name)
            index.add(vectors)
            faiss.write_index(index, index_file_name)

            # Removing metadata file if exists
            if os.path.exists(self.metadata_file):
                os.remove(self.metadata_file)
            with open(self.metadata_file, 'w') as f:
                json.dump(idx_doc, f)

        elif strategy == 'append':
            # Reading existing index file if present
            if os.path.exists(index_file_name):
                index = faiss.read_index(index_file_name)
                existing_idx_doc = {}
                if os.path.exists(self.metadata_file):
                    with open(self.metadata_file, 'r') as f:
                        existing_idx_doc = json.load(f)
                index.add(vectors)
            else:
                index.add(vectors)
                existing_idx_doc = {}

            # Appending new metadata
            existing_idx_doc.update(idx_doc)
            with open(self.metadata_file, 'w') as f:
                json.dump(existing_idx_doc, f)

            faiss.write_index(index, index_file_name)

        else:
            raise ValueError("Invalid strategy. Use 'overwrite' or 'append'.")

        # Verifying that index and metadata files have the same number of records
        index_size = faiss.read_index(index_file_name).ntotal
        with open(self.metadata_file, 'r') as f:
            idx_doc_size = len(json.load(f))
        
        if index_size != idx_doc_size:
            raise ValueError("Mismatch between number of records in the index file and metadata file.")

        # Logging the completion message
        logging.info(f"Indexed {len(docs)} documents (type={index_type}) and stored embeddings in '{index_file_name}' using '{strategy}' strategy.")
        logging.info(f"Metadata saved to '{self.metadata_file}'.")

class Retriever:
    def __init__(self, db: FAISSDB):
        """Initializing Retriever with a reference to the FAISSDB object."""
        self.db = db
        self.context_with_metadata = []

    def retrieve(self, query: str, index_file_name='faiss.index', top_k: int = 5) -> List[Document]:
        """Searching the FAISS index for similar vectors and retrieving documents.

        Args:
            query: The query string to search for.
            index_file_name: The name of the index file.
            top_k: The number of nearest neighbors to retrieve.

        Returns:
            A list of Document objects corresponding to the top_k nearest neighbors.
        """
        # Disable all logging
        logging.disable(logging.CRITICAL)
        # Encoding query to embedding
        query_embedding = self.db.model.encode([query], device=device, show_progress_bar=False).astype(np.float32)
        index = faiss.read_index(index_file_name)
        distances, indices = index.search(query_embedding, top_k)

        self.db.metadata_file = f"{index_file_name}_metadata.json"
        # Loading metadata from file
        if os.path.exists(self.db.metadata_file):
            with open(self.db.metadata_file, 'r') as f:
                idx_doc = json.load(f)
        else:
            idx_doc = {}

        # Retrieving Document objects for the search results
        search_results = []
        another_output = []
        for idx in indices[0]:
            doc_info = idx_doc.get(str(idx), {})  # Fetching metadata based on document ID
            if doc_info:
                _content = doc_info.get("content", "")
                _metadata= doc_info.get("metadata", {})
                document = Document(
                    page_content=_content,
                    metadata=_metadata
                )
                search_results.append(document)
                
                metadata = ', '.join(f"{key}={value}" for key, value in _metadata.items() if value)

                another_output.append({'content':_content,'metadata':metadata})
        
        self.context_with_metadata = another_output
        
        return search_results

    def context(self, query: str, index_file_name='faiss.index', top_k: int = 7) -> str:
        """Returns concatenated content of the top_k nearest neighbors.

        Args:
            query: The query string to search for.
            index_file_name: The name of the index file.
            top_k: The number of nearest neighbors to retrieve.

        Returns:
            A concatenated string of the contents from the top_k nearest neighbors.
        """
        search_results = self.retrieve(query, index_file_name, top_k)
        return "\n\n".join([doc.page_content for doc in search_results])

# Example usage
# documents = [Document(page_content="Sample text", metadata={}), ...]
# faiss_db = FAISSDB()
# faiss_db.build_index(documents)
# retriever = Retriever(faiss_db)
# results = retriever.retrieve("Sample query")
# print(results)