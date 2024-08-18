import re
import logging
from typing import List

class Document:
    def __init__(self, page_content: str, metadata: dict):
        # Initializing Document with page content and metadata
        self.page_content = page_content
        self.metadata = metadata

def clean_documents(documents: List[Document]) -> List[Document]:
    """Cleaning a list of Document objects by applying text cleaning functions.

    Args:
        documents: A list of Document objects to be cleaned.

    Returns:
        A list of cleaned Document objects.
    """
    def merge_hyphenated_words(text: str) -> str:
        """Merging words split by hyphens and newlines.

        Args:
            text: The text to be cleaned.

        Returns:
            The cleaned text with hyphenated words merged.
        """
        return re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    def fix_newlines(text: str) -> str:
        """Replacing single newlines with spaces to fix text formatting.

        Args:
            text: The text to be cleaned.

        Returns:
            The cleaned text with single newlines replaced by spaces.
        """
        return re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    def remove_multiple_newlines(text: str) -> str:
        """Replacing multiple consecutive newlines with a single newline.

        Args:
            text: The text to be cleaned.

        Returns:
            The cleaned text with multiple newlines reduced to a single newline.
        """
        return re.sub(r"\n{2,}", "\n", text)

    # Defining a list of cleaning functions
    cleaning_functions = [merge_hyphenated_words, fix_newlines]#, remove_multiple_newlines] # commented to preserve paragraphs.

    cleaned_documents = []
    for doc in documents:
        cleaned_text = doc.page_content  # Getting the page content of the document
        for cleaning_function in cleaning_functions:
            cleaned_text = cleaning_function(cleaned_text)  # Applying each cleaning function
        
        cleaned_document = Document(
            page_content=cleaned_text,  # Creating a new Document object with cleaned text
            metadata=doc.metadata
        )
        cleaned_documents.append(cleaned_document)  # Adding the cleaned document to the list
    
    logging.info(f"Cleaning process completed on {len(documents)} documents.")  # Logging completion
    return cleaned_documents
