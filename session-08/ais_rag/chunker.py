import logging
from typing import List
from dataclasses import dataclass

# Define the Document class with page content and metadata
@dataclass
class Document:
    page_content: str
    metadata: dict

class RecursiveTextSplitter:
    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 100):
        # Initializing RecursiveTextSplitter with chunk size and overlap
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str, strategy: str = "simple") -> List[str]:
        """Splits text into chunks based on the strategy.

        Args:
            text: The text to be split.
            strategy: The chunking strategy ("simple", "sentence", "paragraph", "character").

        Returns:
            A list of text chunks.
        """
        chunks = []
        if strategy == "simple":
            # Splitting text into chunks based on simple fixed size with overlap
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
                chunks.append(text[i:i + self.chunk_size])
        elif strategy == "sentence":
            # Splitting text into chunks based on sentences
            sentences = text.split(". ")
            current_chunk = ""
            for sentence in sentences:
                if len(current_chunk) + len(sentence) + 1 > self.chunk_size:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence + ". "
                else:
                    current_chunk += sentence + ". "
            if current_chunk:
                chunks.append(current_chunk.strip())
        elif strategy == "paragraph":
            # Splitting text into chunks based on paragraphs
            paragraphs = text.split("\n\n")
            current_chunk = ""
            for paragraph in paragraphs:
                if len(current_chunk) + len(paragraph) + 2 > self.chunk_size:
                    chunks.append(current_chunk.strip())
                    current_chunk = paragraph + "\n\n"
                else:
                    current_chunk += paragraph + "\n\n"
            if current_chunk:
                chunks.append(current_chunk.strip())
        elif strategy == "character":
            # Splitting text into chunks based on characters with flexible separators
            separators = ["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            current_chunk = ""
            while text:
                if len(text) <= self.chunk_size:
                    chunks.append(current_chunk + text)
                    break

                split_pos = -1
                for sep in separators:
                    pos = text.rfind(sep, 0, self.chunk_size)
                    if pos > split_pos:
                        split_pos = pos

                if split_pos == -1:
                    split_pos = self.chunk_size

                split_pos += len(separators[0])  # Including the separator length
                chunk = text[:split_pos]
                text = text[split_pos - self.chunk_overlap:]
                chunks.append(current_chunk + chunk)
                current_chunk = text[:self.chunk_overlap]

            if current_chunk:
                chunks.append(current_chunk.strip())
        else:
            raise ValueError("Invalid chunking strategy")
        return chunks

    def split_documents(self, documents: List[Document], strategy: str = "simple") -> List[Document]:
        """Splits a list of documents into smaller chunks based on the strategy.

        Args:
            documents: The list of Document objects to be split.
            strategy: The chunking strategy ("simple", "sentence", "paragraph", "character").

        Returns:
            A list of Document objects, each containing a chunk of text.
        """
        all_chunks = []
        for doc in documents:
            chunks = self.split_text(doc.page_content, strategy)  # Splitting the document text into chunks
            for i, chunk in enumerate(chunks):
                # Creating a new Document object for each chunk
                chunk_doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": doc.metadata.get("source"),
                        "page_number": doc.metadata.get("page_number"),
                        "chunk": i,
                        **doc.metadata,
                    }
                )
                all_chunks.append(chunk_doc)  # Adding the chunked document to the list
        
        logging.info(f"Converted {len(documents)} documents into {len(all_chunks)} chunks")  # Logging the result
        return all_chunks
