import os
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Get the logger from Phase 1 (assuming it has been configured)
logger = logging.getLogger('RAG_Project')


def chunk_document(text, chunk_size=300, chunk_overlap=50):
    """
    Chunk a document into smaller pieces using LangChain's RecursiveCharacterTextSplitter.

    Note:
        The RecursiveCharacterTextSplitter splits text based on character count, not tokens.
        However, for our purposes we are setting the parameter to 300 to approximate 300 tokens.
        Depending on your tokenizer, you may need to adjust the chunk_size or implement a custom
        token counting function.

    Parameters:
        text (str): The text to be chunked.
        chunk_size (int): The target size of each chunk (default: 300). Adjust as needed.
        chunk_overlap (int): Number of characters to overlap between chunks (default: 50).

    Returns:
        list of str: A list of text chunks.
    """
    # Create an instance of the text splitter. The chunk_size and chunk_overlap are in characters by default.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_text(text)
    logger.debug(
        f"Document split into {len(chunks)} chunks with chunk_size={chunk_size} and chunk_overlap={chunk_overlap}")
    return chunks


def process_all_documents(processed_folder='data/processed', chunks_folder='data/chunks', chunk_size=300,
                          chunk_overlap=50):
    """
    Process all preprocessed documents by chunking their content and saving the chunks.

    Steps:
        1. Read each preprocessed document from the processed folder.
        2. Chunk the content using chunk_document().
        3. Save the resulting chunks into the chunks_folder.

    Parameters:
        processed_folder (str): Folder where processed documents are stored.
        chunks_folder (str): Folder where chunks will be saved.
        chunk_size (int): The target size of each chunk.
        chunk_overlap (int): Overlap between consecutive chunks.

    Returns:
        dict: Mapping of document file names to a list of chunks.
    """
    # Ensure that the chunks_folder exists
    if not os.path.exists(chunks_folder):
        os.makedirs(chunks_folder)
        logger.debug(f"Created chunks folder at: {chunks_folder}")

    chunks_mapping = {}

    # Iterate through each file in the processed folder
    for file_name in os.listdir(processed_folder):
        file_path = os.path.join(processed_folder, file_name)
        # Read the preprocessed text
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            logger.info(f"Chunking document: {file_name}")
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            continue

        # Get chunks for the document
        chunks = chunk_document(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks_mapping[file_name] = chunks

        # Save each chunk into a separate file for clarity, or combine them if preferred.
        # Here, we store all chunks into a single file per document (each chunk separated by a delimiter).
        chunks_file = os.path.join(chunks_folder, file_name + '.chunks.txt')
        try:
            with open(chunks_file, 'w', encoding='utf-8') as out_f:
                for i, chunk in enumerate(chunks):
                    out_f.write(f"--- Chunk {i + 1} ---\n")
                    out_f.write(chunk + "\n\n")
            logger.info(f"Saved {len(chunks)} chunks to {chunks_file}")
        except Exception as e:
            logger.error(f"Error saving chunks for {file_name}: {e}")

    return chunks_mapping


if __name__ == '__main__':
    # Run the document chunking process when the module is executed directly.
    chunks_result = process_all_documents()
    print(
        f"Chunking complete. Processed chunks for {len(chunks_result)} document(s). Check the 'data/chunks' folder and app.log for details.")
