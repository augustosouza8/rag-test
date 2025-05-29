import os
import logging
from PyPDF2 import PdfReader  # Make sure to install PyPDF2 (e.g., pip install PyPDF2)

logger = logging.getLogger('RAG_Project')


def gather_documents(raw_folder='data/raw'):
    """
    Gather document file paths from the raw data folder.

    Searches recursively for files with supported extensions (.txt and .pdf).

    Parameters:
        raw_folder (str): Directory path where raw documents are stored.

    Returns:
        list of str: List of file paths for all gathered documents.
    """
    supported_extensions = ('.txt', '.pdf')
    documents = []
    if not os.path.exists(raw_folder):
        logger.error(f"Raw folder {raw_folder} not found.")
        return documents

    for root, _, files in os.walk(raw_folder):
        for file in files:
            if file.lower().endswith(supported_extensions):
                file_path = os.path.join(root, file)
                documents.append(file_path)
                logger.debug(f"Found document: {file_path}")
    logger.info(f"Gathered {len(documents)} document(s) from {raw_folder}")
    return documents


def extract_text_from_txt(file_path):
    """
    Extract text from a plain text (.txt) file.

    Parameters:
        file_path (str): Path to the text file.

    Returns:
        str: The text content of the file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return text
    except Exception as e:
        logger.error(f"Error reading text file {file_path}: {e}")
        return ""


def extract_text_from_pdf(file_path):
    """
    Extract text from a PDF file using PyPDF2.

    Parameters:
        file_path (str): Path to the PDF file.

    Returns:
        str: The concatenated text extracted from all pages of the PDF.
    """
    try:
        reader = PdfReader(file_path)
        text = ""
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            text += page_text
            logger.debug(f"Extracted text from page {page_num} of {file_path}")
        return text
    except Exception as e:
        logger.error(f"Error reading PDF file {file_path}: {e}")
        return ""


def preprocess_text(text):
    """
    Preprocess the extracted text.

    Currently performs basic cleaning, such as stripping leading/trailing whitespace.
    Additional preprocessing (e.g., normalization, stop-word removal) can be added if needed.

    Parameters:
        text (str): Raw text extracted from a document.

    Returns:
        str: The preprocessed text.
    """
    return text.strip()


def process_documents(raw_folder='data/raw', processed_folder='data/processed'):
    """
    Process documents by gathering them, extracting text, preprocessing the content,
    and saving the processed output in the designated folder.

    Parameters:
        raw_folder (str): Path to the folder containing raw documents.
        processed_folder (str): Path to the folder where preprocessed documents will be saved.

    Returns:
        dict: A mapping of original file names to their preprocessed text content.
    """
    documents = gather_documents(raw_folder)
    processed_data = {}

    # Ensure the processed folder exists
    if not os.path.exists(processed_folder):
        os.makedirs(processed_folder)
        logger.debug(f"Created processed folder at: {processed_folder}")

    for file_path in documents:
        file_name = os.path.basename(file_path)
        logger.info(f"Processing document: {file_name}")

        if file_name.lower().endswith('.txt'):
            raw_text = extract_text_from_txt(file_path)
        elif file_name.lower().endswith('.pdf'):
            raw_text = extract_text_from_pdf(file_path)
        else:
            logger.warning(f"Unsupported file type for {file_name}. Skipping.")
            continue

        preprocessed_text = preprocess_text(raw_text)
        processed_data[file_name] = preprocessed_text

        # Save the preprocessed text to the processed folder
        processed_file_path = os.path.join(processed_folder, file_name + '.processed.txt')
        try:
            with open(processed_file_path, 'w', encoding='utf-8') as f:
                f.write(preprocessed_text)
            logger.info(f"Saved preprocessed document to {processed_file_path}")
        except Exception as e:
            logger.error(f"Error saving processed file {processed_file_path}: {e}")

    return processed_data


if __name__ == '__main__':
    # Execute document processing if this module is run directly.
    processed_docs = process_documents()
    print(f"Processed {len(processed_docs)} documents. Check app.log for details.")
